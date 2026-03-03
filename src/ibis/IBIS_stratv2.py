import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib as mpl
from tqdm import tqdm, tnrange, tqdm_notebook
import dill as pickle
import time
import os
from joblib import Parallel, delayed
import random
import warnings
warnings.filterwarnings("ignore")
import sys
from scipy.signal import savgol_filter
from pathlib import Path

def _enforce_monotone_increasing(x, eps=0.0):
    """Make x nondecreasing; optionally enforce min increment eps."""
    x = np.asarray(x, float).copy()
    x = np.maximum.accumulate(x)
    if eps > 0:
        for i in range(1, x.size):
            if x[i] < x[i-1] + eps:
                x[i] = x[i-1] + eps
    return x

def _safe_pchip(d, a, depth_grid):
    """PCHIP age(depth) with monotone enforcement + finite checks."""
    d = np.asarray(d, float)
    a = np.asarray(a, float)

    ok = np.isfinite(d) & np.isfinite(a)
    d = d[ok]; a = a[ok]
    if d.size < 4:
        return None

    o = np.argsort(d)
    d = d[o]; a = a[o]

    # enforce monotone depth
    d = _enforce_monotone_increasing(d, eps=0.0)

    # remove duplicate depths (keep last)
    m = np.concatenate(([True], np.diff(d) > 0))
    d = d[m]; a = a[m]
    if d.size < 4:
        return None

    # enforce monotone age
    a = _enforce_monotone_increasing(a, eps=0.0)

    f = PchipInterpolator(d, a, extrapolate=False)
    ag = f(depth_grid)
    if np.any(~np.isfinite(ag)):
        return None

    # final monotone guard
    ag = _enforce_monotone_increasing(ag, eps=0.0)
    return ag

def _growth_rate_from_age_grid(depth_grid, age_grid,
                               slope_floor_yr_per_mm=1.0,
                               gr_cap_mm_per_yr=None,
                               smooth_log_gr=True,
                               win=31, poly=2):
    """
    GR(mm/yr) = 1 / (dAge/dDepth), with slope floor to prevent blowups.
    """
    depth_grid = np.asarray(depth_grid, float)
    age_grid   = np.asarray(age_grid, float)

    dage_dd = np.gradient(age_grid, depth_grid)  # yr/mm

    # slope floor (physical max growth => slope_floor = 1/GR_max)
    dage_dd = np.maximum(dage_dd, float(slope_floor_yr_per_mm))

    gr = 1.0 / dage_dd  # mm/yr

    # optional cap (also physical)
    if gr_cap_mm_per_yr is not None:
        gr = np.clip(gr, 0.0, float(gr_cap_mm_per_yr))

    # optional smooth in log space
    if smooth_log_gr:
        ok = np.isfinite(gr) & (gr > 0)
        if ok.sum() >= max(win, 9):
            w = int(win)
            if w % 2 == 0:
                w += 1
            w = min(w, ok.sum() if ok.sum() % 2 == 1 else ok.sum() - 1)
            if w >= 9:
                lg = np.log(gr[ok])
                lg_s = savgol_filter(lg, window_length=w, polyorder=min(poly, w-1))
                gr2 = gr.copy()
                gr2[ok] = np.exp(lg_s)
                gr = gr2

    return gr

def _logspace_percentiles(G, qs=(2.5, 50, 97.5), eps=1e-12):
    """
    Percentiles in log space (robust for positive, skewed rates).
    """
    G = np.asarray(G, float)
    Gc = np.clip(G, eps, np.inf)
    lg = np.log(Gc)
    P = np.percentile(lg, qs, axis=0)
    return tuple(np.exp(Pi) for Pi in P)



class IBIS_Strat:
    """
    IBIS_Strat
    ----------
    - Make an Age-depth model
    - Input the U-Th ages and 68% credible levels from the IBIS_MCMC_Initial Thorium code.
    - This code must be run first, if it is there will be no file to read and it should go back and run part1 of the code before making an age-depth model
    """
    def __init__(self, U_series_ages,
                 U_series_ages_err_low, U_series_ages_err_high,
                 data, sample_name='SAMPLE_NAME',
                 Start_from_pickles=True, n_chains=3,
                 iterations=50000, burn_in=10000,
                 Top_Age_Stal=False, thin=10,
                 store_thin=10,   # <-- ADD THIS

                 resolution=100,
             ceiling_only=True,
             simple_bounds=True,
             pad_frac=0.20,
             save_dir = None):
        """
        Parameters
        ----------
        resolution : int
            Number of nodes used for the latent age-depth curve.
        bounding : bool
            If True, add synthetic top/bottom guard points (Chron-style).
        pad_frac : float
            Fraction of depth range to pad when bounding=True.
        """
        # --- core config ---
        self.sample_name = str(sample_name)
        self.Start_from_pickles = bool(Start_from_pickles)
        self.n_chains = int(n_chains)
        self.iterations = int(iterations)
        self.burn_in = int(burn_in)
        self.thin = int(thin) if thin and thin > 0 else 1
        self.store_thin = int(store_thin) if store_thin and store_thin > 0 else 10

        self.Top_Age_Stal = bool(Top_Age_Stal)
        self.resolution = int(resolution)
        self.pad_frac = float(pad_frac)
        
        # Save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()
        self.save_dir.mkdir(parents = True, exist_ok = True)


        # --- data (copy so caller arrays remain untouched) ---
        U_series_ages = np.asarray(U_series_ages, float).copy()
        U_series_ages_err_low = np.asarray(U_series_ages_err_low, float).copy()
        U_series_ages_err_high = np.asarray(U_series_ages_err_high, float).copy()
        # ---- Depths (prefer Depth_Meas, else Depth) ----
        if "Depth_Meas" in data.columns:
            self.Depths = np.asarray(data["Depth_Meas"].values, float).copy()
            depth_col_used = "Depth_Meas"
        elif "Depths" in data.columns:
            self.Depths = np.asarray(data["Depths"].values, float).copy()
            depth_col_used = "Depth"
        else:
            raise KeyError(f"No depth column found. Expected 'Depth_Meas' or 'Depth'. "
                        f"Available columns: {list(data.columns)}")
        
        # ---- Depth uncertainties (optional; default to zeros if missing) ----
        if "Depth_Meas_err" in data.columns:
            self.Depths_err = np.asarray(data["Depth_Meas_err"].values, float).copy()
            depth_err_col_used = "Depth_Meas_err"
        elif "Depths_err" in data.columns:
            self.Depths_err = np.asarray(data["Depths_err"].values, float).copy()
            depth_err_col_used = "Depth_err"
        else:
            self.Depths_err = np.zeros_like(self.Depths)
            depth_err_col_used = None
        
        # Optional top stal bound (young minimum)
        # Here if the cave if active - still dripping speleothem want to allow the
        # user to input a zero age at the top to define this, or age of "collection" from present day of running to collection
        
        if self.Top_Age_Stal:
            U_age_top = 0.0
            U_age_top_err = 0.1
            self.u_ages = np.insert(U_series_ages, 0, U_age_top)
            self.low_age_err = np.insert(U_series_ages_err_low, 0, U_age_top_err) / 2.0
            self.high_age_err = np.insert(U_series_ages_err_high, 0, U_age_top_err) / 2.0
            self.Depths = np.insert(self.Depths, 0, 0.0)
            self.Depths_err = np.insert(self.Depths_err, 0, 0.01)
        else:
            self.u_ages = U_series_ages
            self.low_age_err = U_series_ages_err_low / 2.0
            self.high_age_err = U_series_ages_err_high / 2.0
            
        self.depth_grid_gr = np.linspace(self.Depths.min(), self.Depths.max(), self.resolution)


        # Floors to avoid log(0)
        EPS = 1e-9
        self.low_age_err = np.maximum(self.low_age_err, EPS)
        self.high_age_err = np.maximum(self.high_age_err, EPS)
        self.Depths_err = np.maximum(self.Depths_err, EPS)
        self.age_floor = 0.0

        # --- simple bounds config ---
        self.simple_bounds = bool(simple_bounds)
        self.ceiling_only  = bool(ceiling_only)

        # compute only a ceiling from deepest datum
        self._compute_ceiling_only()
        self._compute_depth_bounds(k=5.0) # 5𝜎
        # smart-move knobs
        self.p_smart_age        = 0.80   # probability to use the smart age move (vs RW)
        self.prop_scale_mult_age = 2.0   # multiply local sigma_eff for proposal width
        self.neighbor_smooth_bw  = 2      # spread proposed change softly to +/- bw  neighbors (0 = off)
        self.neighbor_smooth_frac = 0.6   # how much of the change is shared to neighbors
        

    def _min_sep_value(self):
        """
        Minimum allowed spacing between adjacent Depth_Model nodes (in mm).

        Priority:
          1) user-set self.min_sep if it exists and is positive
          2) otherwise infer from measurement depth uncertainty
        """
        ms = getattr(self, "min_sep", None)
        if ms is not None and float(ms) > 0:
            return float(ms)

        # conservative default: a fraction of typical depth uncertainty
        # (tune the 0.25 up/down; 0.25–1.0 are common)
        dep_err = np.asarray(self.Depths_err, float)
        base = np.nanmedian(dep_err[np.isfinite(dep_err)]) if dep_err.size else np.nan
        if not np.isfinite(base) or base <= 0:
            base = 1e-3  # fallback mm
        return max(0.25 * base, 1e-6)


    def _enforce_min_sep(self, d, min_sep, floor=0.0, ceiling=np.inf):
        """
        Enforce monotone increasing depths with minimum spacing `min_sep`.
        Keeps the order (does not re-permute).
        """
        d = np.asarray(d, float).copy()

        # clamp to bounds first
        if np.isfinite(floor):
            d = np.maximum(d, float(floor))
        if np.isfinite(ceiling):
            d = np.minimum(d, float(ceiling))

        # forward pass: enforce d[i] >= d[i-1] + min_sep
        for i in range(1, d.size):
            need = d[i - 1] + float(min_sep)
            if d[i] < need:
                d[i] = need

        # if we blew past ceiling, pull back with a backward pass
        if np.isfinite(ceiling) and d.size > 0 and d[-1] > float(ceiling):
            d[-1] = float(ceiling)
            for i in range(d.size - 2, -1, -1):
                need = d[i + 1] - float(min_sep)
                if d[i] > need:
                    d[i] = need

            # re-apply forward to fix any floor-side issues
            if np.isfinite(floor) and d.size > 0 and d[0] < float(floor):
                d[0] = float(floor)
            for i in range(1, d.size):
                need = d[i - 1] + float(min_sep)
                if d[i] < need:
                    d[i] = need

        return d


            
    def _compute_ceiling_only(self):
        """Compute only an upper (too-old) age ceiling from the deepest observation."""
        i_bot = int(np.argmax(self.Depths))  # deepest datum
        sig_bot = 0.5 * (self.low_age_err[i_bot] + self.high_age_err[i_bot])
        bot_age = float(self.u_ages[i_bot])
        self.age_ceiling = bot_age + 3.0 * sig_bot
    
    def _compute_depth_bounds(self, k=3.0):
        i_top = int(np.argmin(self.Depths))
        i_bot = int(np.argmax(self.Depths))
        sig_top = float(self.Depths_err[i_top])
        sig_bot = float(self.Depths_err[i_bot])

        raw_floor = float(self.Depths[i_top]) - k * sig_top
        raw_ceil  = float(self.Depths[i_bot]) + k * sig_bot

        # hard non-negative depth floor
        self.depth_floor   = max(0.0, raw_floor)
        self.depth_ceiling = max(self.depth_floor + 1e-9, raw_ceil)  # `keep valid box

        
    # ---------- helpers: bounding / guards ----------
    def _apply_simple_bound_guards(self):
        """Insert two guard observations at min/max depth near the hard     bounds."""
        dmin, dmax = float(np.min(self.Depths)), float(np.max(self.Depths))
        i_top = int(np.argmin(self.Depths))
        i_bot = int(np.argmax(self.Depths))
    
        # end-member symmetric sigmas
        sig_top = 0.5 * (self.low_age_err[i_top] + self.high_age_err[i_top])
        sig_bot = 0.5 * (self.low_age_err[i_bot] + self.high_age_err[i_bot])
    
        # place guards slightly inside bounds if epsilon given, else exactly at     bound
        eps_top = (self.bounds_epsilon if self.bounds_epsilon is not None else  0.0)
        eps_bot = (self.bounds_epsilon if self.bounds_epsilon is not None else  0.0)
    
        guard_top_age = self.age_floor + eps_top
        guard_bot_age = self.age_ceiling - eps_bot
    
        # small but non-zero sigmas for the guards
        gsig_top = max(1e-6, self.guard_sigma_scale * sig_top)
        gsig_bot = max(1e-6, self.guard_sigma_scale * sig_bot)
    
        # prepend/append
        self.u_ages       = np.r_[guard_top_age, self.u_ages,           guard_bot_age]
        self.low_age_err  = np.r_[gsig_top,      self.low_age_err,  gsig_bot]
        self.high_age_err = np.r_[gsig_top,      self.high_age_err, gsig_bot]
        self.Depths       = np.r_[dmin,          self.Depths,       dmax]
        self.Depths_err   = np.r_[1e-9,          self.Depths_err,   1e-9]
    
    # ---------- priors ----------
    def _rw2_logprior(self, age, lam=1e-6):
        """
        Prior on curvature of the age-depth relationship
        For a speleothem I think this should be fairly weak
        """
        d1 = np.diff(age)
        if d1.size < 2:
            return 0.0
        d2 = np.diff(d1)
        return float(-0.5 * lam * np.dot(d2, d2))


    def Log_Priors(self, theta):
        Age_Model, Depth_Model = theta
        if np.min(Age_Model) < self.age_floor:
            return -np.inf
        if np.min(Depth_Model) < self.depth_floor:
            return -np.inf

        if self.simple_bounds:
            # age ceiling only: forbid models that get too old anywhere
            if np.max(Age_Model) > self.age_ceiling:
                return -np.inf
            # depth box: keep nodes inside [depth_floor, depth_ceiling]
            if (np.min(Depth_Model) < self.depth_floor) or (np.max(Depth_Model) > self.depth_ceiling):
                return -np.inf

                
        # monotone age prior (reject if violated)
        if np.any(np.diff(Age_Model) < 0.0):
            return -np.inf

        lp = 0.0
        # broad uniforms priors on Age - STAY IN BOUNDS
        lp += np.sum(uniform.logpdf(Age_Model, loc=0.0, scale=1e6))
        lp += np.sum(uniform.logpdf(Depth_Model,
                                    loc=self.depth_floor,
                                    scale=max(self.depth_ceiling - self.depth_floor, 1.0)))

        # Weak smoothness
        lp += self._rw2_logprior(Age_Model, lam=1e-8)
        return float(lp)

    def Initial_Guesses_for_Model(self):
        initial_thetas = []
        # linear fit seed (Chron-like)
        A = np.vstack([np.ones_like(self.Depths), self.Depths]).T
        b0, b1 = np.linalg.lstsq(A, self.u_ages, rcond=None)[0]
        # existing:
        Depth_Model0 = np.linspace(self.Depths.min(), self.Depths.max(), self.resolution)
        Age_Model0 = b0 + b1 * Depth_Model0
        Age_Model0 = np.clip(Age_Model0, self.age_floor + 1e-12, self.age_ceiling - 1e-12)
        Depth_Model0 = np.clip(Depth_Model0, self.depth_floor + 1e-12, self.depth_ceiling - 1e-12)

        # keep seeds within boxes (tiny margin to avoid sitting exactly on edge)
        if self.simple_bounds:
            epsA = 1e-9
            epsD = 1e-9
            # ceiling only for age
            Age_Model0   = np.minimum(Age_Model0, self.age_ceiling - epsA)
            # full box for depth
            Depth_Model0 = np.minimum(np.maximum(Depth_Model0, self.depth_floor + epsD),
                                      self.depth_ceiling - epsD)

        for _ in range(self.n_chains):
            # small jitter around the linear seed
            Age_Model = Age_Model0 + np.random.normal(0.0, np.nanmean(self.high_age_err) * 0.25, size=Age_Model0.size)
            Depth_Model = Depth_Model0.copy()
            theta_initial = (Age_Model, Depth_Model)
            if np.isfinite(self.Log_Priors(theta_initial)):
                initial_thetas.append(theta_initial)
            else:
                initial_thetas.append((Age_Model0.copy(), Depth_Model0.copy()))
        return initial_thetas

    def Logpdf_asymm(self, mu, sigma_low, sigma_high, model):
        """
        Asymmetric normal log-likelihood:
        - if (mu - model) < 0  -> use sigma_low
        - else                  -> use sigma_high
        All inputs are 1-D arrays of the same length.
        
        """
        mu = np.asarray(mu, float)
        model = np.asarray(model, float)
        sigma_low = np.asarray(sigma_low, float)
        sigma_high = np.asarray(sigma_high, float)

        delta = mu - model
        sigma = np.where(delta < 0.0, sigma_low, sigma_high)
        # tiny floor for safety
        sigma = np.maximum(sigma, 1e-12)

        ll = -0.5 * (delta / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
        return float(np.sum(ll))

    def _interp_model_to_obs(self, Depth_Model, Age_Model):
        # assumes Depth_Model monotone increasing
        return np.interp(self.Depths, Depth_Model, Age_Model)

    def Log_Likelihood(self, theta):
        Age_Model, Depth_Model = theta
    
        # interpolate age curve to observation depths
        age_at_obs = self._interp_model_to_obs(Depth_Model, Age_Model)
    
        # propagate depth error to age domain via slope
        slope = np.gradient(Age_Model, Depth_Model)
        slope_at_obs = np.interp(self.Depths, Depth_Model, slope)
        sigma_depth_age = np.abs(slope_at_obs) * self.Depths_err
    
        # build side-specific effective sigmas
        sigma_low_eff  = np.sqrt(self.low_age_err**2  + sigma_depth_age**2 +    1e-12)
        sigma_high_eff = np.sqrt(self.high_age_err**2 + sigma_depth_age**2 +    1e-12)
    
        # true asymmetric likelihood with depth→age folded in
        return self.Logpdf_asymm(self.u_ages, sigma_low_eff, sigma_high_eff,    age_at_obs)
    
    # ---------- posterior ----------
    def Log_Posterior(self, theta):
        lp = self.Log_Priors(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.Log_Likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)
        
    def _interp_obs_age_at_depth(self, d):
        """Interpolated observed age at depth d (linear)."""
        return float(np.interp(d, self.Depths, self.u_ages))

    def _sigma_eff_sym_at_depth(self, depth, Age_Model, Depth_Model):
        """
        Symmetric effective sigma at 'depth':
        combine measurement (mean of low/high) with depth->age propagation via slope.
        """
        # slope on model grid
        slope = np.gradient(Age_Model, Depth_Model)
        slope_at_d = float(np.interp(depth, Depth_Model, slope))
        # obs sigma at nearest observed depth (linear interp)
        sig_low  = float(np.interp(depth, self.Depths, self.low_age_err))
        sig_high = float(np.interp(depth, self.Depths, self.high_age_err))
        sig_meas = 0.5 * (sig_low + sig_high)
        sig_depth = abs(slope_at_d) * float(np.interp(depth, self.Depths, self.Depths_err))
        return max(1e-12, np.hypot(sig_meas, sig_depth))
    
    # ---------- proposals ----------
    def Age_Depth_Model_move(self, theta, tuning_factor, index):
        Age_Model, Depth_Model = theta
        Age_Model_prime  = Age_Model.copy()
        Depth_Model_prime = Depth_Model.copy()
        log_post_cur = self.Log_Posterior(theta)
    
        rng = np.random.rand()
    
        # -------- Depth move ----------
        if rng < 0.05:
            k = max(1, int(0.05 * Depth_Model_prime.size))
            idxs = np.random.choice(Depth_Model_prime.size,
            size=k,
            replace=False)
            Depth_Model_prime[idxs] += np.random.normal(0.0,
            np.median(self.Depths_err), size=k)
            
            if self.simple_bounds:
                Depth_Model_prime[idxs] = np.clip(Depth_Model_prime[idxs],
                 self.depth_floor, self.depth_ceiling)
            # enforce monotone depths
            perm = np.argsort(Depth_Model_prime)
            Depth_Model_prime = Depth_Model_prime[perm]
            Age_Model_prime   = Age_Model_prime[perm]

            # enforce minimum spacing between adjacent depth nodes
            min_sep = self._min_sep_value()
            floor = self.depth_floor if self.simple_bounds else 0.0
            ceiling = self.depth_ceiling if self.simple_bounds else np.inf
            Depth_Model_prime = self._enforce_min_sep(Depth_Model_prime, min_sep, floor=floor, ceiling=ceiling)

    
        # -------- Smart AGE move --------
        elif rng < 0.05 + self.p_smart_age:
            d_j  = float(Depth_Model_prime[index])
            mu_j = self._interp_obs_age_at_depth(d_j)
            sd_j = self._sigma_eff_sym_at_depth(d_j, Age_Model_prime,   Depth_Model_prime)
            prop_sd = max(1e-8, self.prop_scale_mult_age * sd_j)
    
            a_old = float(Age_Model_prime[index])
            a_new = float(np.random.normal(mu_j, prop_sd))
            a_new = max(a_new, self.age_floor + 1e-12)
            # apply age ceiling-only box, if requested
            if self.simple_bounds and hasattr(self, 'age_ceiling'):
                a_new = min(a_new, float(self.age_ceiling) - 1e-12)
    
            delta = a_new - a_old
            Age_Model_prime[index] = a_new
    
            # Share out amongst neighbors
            bw   = int(self.neighbor_smooth_bw)
            frac = float(self.neighbor_smooth_frac)
            if bw > 0 and frac > 0.0:
                n = Age_Model_prime.size
                for off in range(1, bw + 1):
                    w = frac * np.exp(-0.5 * (off / max(1e-9, bw))**2)  #   Gaussian-ish taper
                    if index - off >= 0:
                        Age_Model_prime[index - off] += w * delta
                    if index + off < n:
                        Age_Model_prime[index + off] += w * delta
    
            # project back to monotone (non-decreasing) with a simple clamp
            # forward pass
            for i in range(index + 1, Age_Model_prime.size):
                if Age_Model_prime[i] < Age_Model_prime[i - 1]:
                    Age_Model_prime[i] = Age_Model_prime[i - 1]
            # backward pass
            for i in range(index - 1, -1, -1):
                if Age_Model_prime[i] > Age_Model_prime[i + 1]:
                    Age_Model_prime[i] = Age_Model_prime[i + 1]
    
        # -------- Plain AGE random walk (fallback) --------
        else:
            Age_Model_prime[index] += float(np.random.normal(0.0, tuning_factor))

            # clamp to [age_floor, age_ceiling]
            Age_Model_prime[index] = max(Age_Model_prime[index], self.age_floor + 1e-12)
            if self.simple_bounds and hasattr(self, 'age_ceiling'):
                Age_Model_prime[index] = min(Age_Model_prime[index], float(self.age_ceiling) - 1e-12)

            # clamp to monotone
            for i in range(index + 1, Age_Model_prime.size):
                if Age_Model_prime[i] < Age_Model_prime[i - 1]:
                    Age_Model_prime[i] = Age_Model_prime[i - 1]
            for i in range(index - 1, -1, -1):
                if Age_Model_prime[i] > Age_Model_prime[i + 1]:
                    Age_Model_prime[i] = Age_Model_prime[i + 1]
    
        # MH accept/reject
        theta_prime = (Age_Model_prime, Depth_Model_prime)
        log_post_prop = self.Log_Posterior(theta_prime)
        log_alpha = log_post_prop - log_post_cur
        accept = (log_alpha >= 0.0) or (np.log(np.random.rand()) < log_alpha)

    
        return (Age_Model_prime if accept else Age_Model,
                Depth_Model_prime if accept else Depth_Model,
                bool(accept))
    

    @staticmethod
    def update_Age_Depth_Model(theta, new_age, new_depth):
        return (new_age, new_depth)

    # ---------- Paths for PICKLES ----------
    def _theta_path(self, chain_id):
        return f"{self.sample_name}_Age_Depth_theta_{chain_id}.pkl"

    def _tuning_path(self, chain_id):
        return f"{self.sample_name}_tuning_factor_Age_Depth_{chain_id}.pkl"

    def _meta_path(self):
        return f"{self.sample_name}_runmeta.pkl"
        


    
    def _growth_rate_on_depth_grid(self, Age_Model, Depth_Model, depth_grid, eps=1e-12,
                                smooth=True, win=21, poly=2):
        """
        Compute GR(depth) = dDepth/dAge from a monotone Age(depth) curve.
        Returns GR on the provided depth_grid (same units as Depth_Model per Age_Model unit).
        """
        a = np.asarray(Age_Model, float)
        d = np.asarray(Depth_Model, float)
    
        # enforce monotone depth and age
        d = np.maximum.accumulate(np.clip(d, 0.0, None))
        a = np.maximum.accumulate(np.clip(a, 0.0, None))
    
        p = np.argsort(d)
        d = d[p]; a = a[p]
    
        # interpolate age onto depth_grid
        ag = np.interp(depth_grid, d, a)
    
        # slope da/dd on depth_grid
        da_dd = np.gradient(ag, depth_grid)
    
        # growth rate = 1 / (da/dd)
        gr = 1.0 / np.maximum(da_dd, eps)
    
        # optional: smooth log(gr) to avoid spikes
        gr = np.asarray(gr, float)
        bad = ~np.isfinite(gr) | (gr <= 0)
        gr[bad] = np.nan
    
        if smooth:
            ok = np.isfinite(gr)
            if ok.sum() >= max(win, 5):
                # make window odd and <= ok.sum()
                w = int(win)
                if w % 2 == 0:
                    w += 1
                w = min(w, ok.sum() if ok.sum() % 2 == 1 else ok.sum() - 1)
                if w >= 5:
                    lg = np.log(gr[ok])
                    lg_s = savgol_filter(lg, window_length=w, polyorder=min(poly, w-1))
                    gr_s = np.exp(lg_s)
                    gr2 = gr.copy()
                    gr2[ok] = gr_s
                    gr = gr2
    
        return gr
    
    def _log_slope_on_depth_grid(self, Age_Model, Depth_Model, depth_grid, eps=1e-12,
                                 smooth=True, win=21, poly=2):
        """
        Return log(dAge/dDepth) on depth_grid. This is the stable object to summarize.
        """
        a = np.asarray(Age_Model, float)
        d = np.asarray(Depth_Model, float)

        d = np.maximum.accumulate(np.clip(d, 0.0, None))
        a = np.maximum.accumulate(np.clip(a, 0.0, None))

        p = np.argsort(d)
        d, a = d[p], a[p]

        ag = np.interp(depth_grid, d, a)
        da_dd = np.gradient(ag, depth_grid)
        da_dd = np.maximum(da_dd, eps)

        logS = np.log(da_dd)

        if smooth:
            ok = np.isfinite(logS)
            if ok.sum() >= max(win, 5):
                w = int(win)
                if w % 2 == 0:
                    w += 1
                w = min(w, ok.sum() if ok.sum() % 2 == 1 else ok.sum() - 1)
                if w >= 5:
                    logS2 = logS.copy()
                    logS2[ok] = savgol_filter(logS[ok], window_length=w, polyorder=min(poly, w-1))
                    logS = logS2

        return logS


    # ---------- MCMC driver ----------
    def MCMC(self, theta, niters, chain_id):
        Age_Model, Depth_Model = theta
        if Age_Model.size == 0 or Depth_Model.size == 0:
            raise ValueError("Initial models cannot be empty")

        Nmodel = Age_Model.size
        target = 0.234

        # load/init per-index tuning
        tuning_file = self._tuning_path(chain_id)
        if os.path.exists(tuning_file) and self.Start_from_pickles:
            with open(tuning_file, 'rb') as f:
                tuning_factors = pickle.load(f)
        else:
            base = float(np.mean(self.high_age_err))
            tuning_factors = {f'Age_Depth_Model_Z_{i}': max(base, 1e-3) for i in range(Nmodel)}

        niters = int(niters)
        burn = int(self.burn_in)
        thin_store = int(getattr(self, "store_thin", 10))
        
        # how many kept?
        if niters <= 1 or niters - 1 <= burn:
            n_keep = 0
        else:
            n_keep = 1 + ((niters - 1 - burn) // thin_store)
        
        # allocate kept draws
        Age_store = np.empty((n_keep, Nmodel)) if n_keep > 0 else np.empty((0, Nmodel))
        Dep_store = np.empty((n_keep, Nmodel)) if n_keep > 0 else np.empty((0, Nmodel))
        post_store = np.empty((n_keep,)) if n_keep > 0 else np.empty((0,))
        
        depth_grid = np.linspace(self.Depths.min(), self.Depths.max(), self.resolution)
        Ngr = depth_grid.size
        Gr_store = np.empty((n_keep, Ngr)) if n_keep > 0 else np.empty((0, Ngr))

        # acceptance accounting
        proposal_counts = {k: 0 for k in tuning_factors}
        accept_counts = {k: 0 for k in tuning_factors}
        total_props = 0
        total_accs = 0

        # checkpoint start
        thetapath_full = self.save_dir / self._theta_path(chain_id)
        tuningpath_full = self.save_dir / self._tuning_path(chain_id)
        with open(thetapath_full, 'wb') as f:
            pickle.dump((Age_Model, Depth_Model), f)
        with open(tuningpath_full, 'wb') as f:
            pickle.dump(tuning_factors, f)

        save_idx = 0
        pbar = tqdm(range(1, niters),
        desc=f"Chain {chain_id}",
        total=niters - 1,
        leave=False,
        dynamic_ncols=True,
        ncols = 100,
        disable = (chain_id != 0) or sys.stdout.isatty())
        FLUSH_EVERY = max(self.thin, 1000)

        for i in pbar:
            key = f'Age_Depth_Model_Z_{np.random.randint(0, Nmodel)}'
            step_scale = tuning_factors[key]
            idx = int(key.split('_')[-1])

            new_age, new_dep, accepted = self.Age_Depth_Model_move((Age_Model, Depth_Model), step_scale, idx)

            proposal_counts[key] += 1
            total_props += 1
            if accepted:
                accept_counts[key] += 1
                total_accs += 1
                Age_Model, Depth_Model = self.update_Age_Depth_Model((Age_Model, Depth_Model), new_age, new_dep)

            # --- Adaption Utilities for the burn-in  ---
            adapt_interval = 200       # how often
            window_min     = 20        # Least number of accepted proposals
            strength       = 2.0       # Strenght of tuning update

            if (i < burn) and (i % adapt_interval == 0):
                # Decay of learning rate
                eta = 0.05 + 0.50 * (1.0 - (i / max(burn, 1)))  # in (0.05, 0.55]
                for k in range(Nmodel):
                    kkey = f'Age_Depth_Model_Z_{k}'
                    p = proposal_counts[kkey]; a = accept_counts[kkey]
                    if p >= window_min:
                        acc = a / p  # per-parameter acceptance in this window

                        # Update of tuning parameters - push toward desired acceptance ratexf
                        log_scale = np.log(tuning_factors[kkey])
                        log_scale += eta * strength * (acc - target)   # push toward target
                        tuning = float(np.exp(log_scale))

                        # Help if things go a bit wacky.
                        if acc > 0.70:
                            tuning *= 2.0
                        elif acc < 0.10:
                            tuning *= 0.5

                        # Clamp and store
                        tuning_factors[kkey] = float(np.clip(tuning, 1e-8, 1e8))

                    # reset window counters
                    proposal_counts[kkey] = 0
                    accept_counts[kkey]   = 0

            # keep thinned post-burn draws (STORE THINNING)
            if (i >= burn) and (((i - burn) % thin_store) == 0) and (save_idx < n_keep):
                Age_store[save_idx, :] = Age_Model
                Dep_store[save_idx, :] = Depth_Model

                logS = self._log_slope_on_depth_grid(
                    Age_Model, Depth_Model, depth_grid, eps=1e-12, smooth=True
                )
                Gr_store[save_idx, :] = logS

                post_store[save_idx] = self.Log_Posterior((Age_Model, Depth_Model))
                save_idx += 1
            
            # Display progres bar stuff
            if i % niters == 0:
                phase = "burn_in" if i < self.burn_in else "sampling"
                props = sum(self.proposal_counts.values()) or 1
                accs = sum(self.accept_counts.values())
                acc_rate = accs / props
                pbar.set_postfix_str(f"phase, acc = {acc_rate:.3f}")

            # periodic checkpoint
            if (i % 5000) == 0:
                thetapath_full = self.save_dir / self._theta_path(chain_id)
                tuningpath_full = self.save_dir / self._tuning_path(chain_id)
                with open(thetapath_full, 'wb') as f:
                    pickle.dump((Age_Model, Depth_Model), f)
                with open(tuningpath_full, 'wb') as f:
                    pickle.dump(tuning_factors, f)

        # final checkpoint
        thetapath_full = self.save_dir / self._theta_path(chain_id)
        tuningpath_full = self.save_dir / self._tuning_path(chain_id)
        with open(thetapath_full, 'wb') as f:
            pickle.dump((Age_Model, Depth_Model), f)
        with open(tuningpath_full, 'wb') as f:
            pickle.dump(tuning_factors, f)

        return Age_store, Dep_store, Gr_store, post_store


    # ---------- SEED Everything ----------
    def check_starting_parameters(self):
        thetas = []
        for chain_id in range(self.n_chains):
            pathfull = self.save_dir / self._theta_path(chain_id)
            if self.Start_from_pickles and pathfull.exists():
                with open(pathfull, 'rb') as f:
                    thetas.append(pickle.load(f))
            else:
                thetas.append(None)

        if any(t is None for t in thetas):
            thetas = self.Initial_Guesses_for_Model()
            print('Starting from linear-fit guesses')
        else:
            print(f'Loaded {self.n_chains} chains from pickles (Start_from_pickles={self.Start_from_pickles})')
        return thetas
        
    
    # MCMC IS RUN FROM HERE
    def Run_MCMC_Strat(self):
        all_thetas = self.check_starting_parameters()
        chain_ids = range(self.n_chains)

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs=-1)(
            delayed(run_chain)(theta, chain_id)
            for theta, chain_id in zip(all_thetas[:self.n_chains], chain_ids)
        )
        self.Chain_Results = results

        # save run meta
        meta = dict(
            sample_name=self.sample_name,
            iterations=self.iterations, burn_in=self.burn_in, thin=self.thin,
            resolution=self.resolution,
            pad_frac=self.pad_frac,
            n_chains=self.n_chains
        )
        metapath_full = self.save_dir / self._meta_path()
        with open(metapath_full, 'wb') as f:
            pickle.dump(meta, f)

        return self.Chain_Results

    def Ensure_Chain_Results(self):
        if self.Chain_Results is None:
            self.Chain_Results = self.Run_MCMC_Strat()  # fixed method name
        return self.Chain_Results

    # ---------- postprocessing - get the results dictionary for all other stuff ----------
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================

    def Get_Results_Dictionary(self):
        if self.Chain_Results is None:
            self.Run_MCMC_Strat()
        Results_ = self.Chain_Results

        z_vars = [f"z{i+1}" for i in range(4)]
        results_dict = []
        for chain_id, result in enumerate(Results_, start=1):
            rd = {}
            for var_name, value in zip(z_vars, result):
                rd[f"{var_name}_{chain_id}"] = value
            results_dict.append(rd)
        return results_dict


    def Get_Posterior_plot(self):

        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]  # get the dictionary for chain i
            log_p.append(chain_dict[f"z4_{i}"])

        fig, ax = plt.subplots(1,1, figsize = (5,5))

        for i in range(self.n_chains):
            ax.plot(log_p[i],
               label = f'Chain {i + 1}')
    
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Posterior')
        ax.set_xscale('log')
        ax.legend(frameon = True, loc = 4, fontsize = 10, ncol = 2)


    def _safe_percentiles_pos(self, X, qs=(2.5, 50, 97.5), method="clip", eps=1e-12):
        X = np.asarray(X, float)

        if method == "clip":
            Xp = np.clip(X, 0.0, None)
            return [np.percentile(Xp, q, axis=0) for q in qs]

        if method == "log1p":
            # safe for zeros: transform -> percentiles -> invert
            Xp = np.clip(X, 0.0, None)
            Y = np.log1p(Xp)
            P = [np.percentile(Y, q, axis=0) for q in qs]
            return [np.expm1(p) for p in P]

        raise ValueError("method must be 'clip' or 'log1p'")


    def Get_Age_Model(self, method="clip"):
        result_dicts = self.Get_Results_Dictionary()
        Age_m = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            Age_m.append(chain_dict[f"z1_{i}"])

        c_age_m = np.concatenate(Age_m)  # (n_samples, n_points)

        low_age, age_median, high_age = self._safe_percentiles_pos(
            c_age_m, qs=(2.5, 50, 97.5), method=method
        )
        return low_age, high_age, age_median


    def Get_Depth_Model(self, method="clip"):
        result_dicts = self.Get_Results_Dictionary()
        Depth_m = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            Depth_m.append(chain_dict[f"z2_{i}"])

        c_Depth_m = np.concatenate(Depth_m)

        low_depth, depth_median, high_depth = self._safe_percentiles_pos(
            c_Depth_m, qs=(2.5, 50, 97.5), method=method
        )
        return low_depth, high_depth, depth_median
        
    def Get_Growth_Rate(self, eps=1e-12, qs=(2.5, 50, 97.5)):
        """
        Robust growth-rate summary from stored Gr_store samples:
          - Work in log(GR) to avoid reciprocal blow-ups.
          - Returns (low95, high95, median) on the SAME grid used in MCMC (depth_grid).
        """
        result_dicts = self.Get_Results_Dictionary()

        GR_all = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            g = np.asarray(chain_dict[f"z3_{i}"], float)   # shape: (n_keep, Ngr)
            GR_all.append(g)

        GR = np.vstack(GR_all)  # (n_samples_total, Ngr)

        # guard + log transform
        GR = np.clip(GR, eps, np.inf)
        logG = np.log(GR)

        qlo, q50, qhi = qs
        log_lo = np.percentile(logG, qlo, axis=0)
        log_md = np.percentile(logG, q50, axis=0)
        log_hi = np.percentile(logG, qhi, axis=0)

        gr_lo = np.exp(log_lo)
        gr_md = np.exp(log_md)
        gr_hi = np.exp(log_hi)

        return gr_lo, gr_hi, gr_md




    def Get_Age_Depth_Plot(self):
        err_asymm = [self.low_age_err * 2,
                    self.high_age_err * 2]

        low_age, high_age, age_med = self.Get_Age_Model(method="clip")   # default
        # or
        low_dep, high_dep, dep_med = self.Get_Depth_Model(method="clip")

        
        fig, ax = plt.subplots(1, 1, figsize = (5,7))
        ax.errorbar(y = self.Depths,
                    x = self.u_ages,
                    xerr = err_asymm,
        fmt = 'o', markerfacecolor = 'dodgerblue',
        markeredgecolor = 'k', ecolor = 'k',
        capsize = 5, label = 'Bayesian\nAge\nEstimate (2$\sigma$)')

        ax.fill_betweenx(dep_med, low_age, high_age,
                         alpha = 0.3, color='navy',
                        label = '95% CI')

        ax.plot(high_age, dep_med,
                         ls = '--', color = 'navy', alpha = 0.4)
        
        ax.plot(low_age, dep_med,
                         ls = '--', color = 'navy', alpha = 0.4)
        
        ax.plot(age_med, dep_med,
                 lw = 1.5, color = 'royalblue', zorder = 2,
                label = 'Median Model')
                
        
        plt.ylim(self.Depths.max() + 2, self.Depths.min() - 2)
        plt.xlabel('Age (a)')
        plt.ylabel('Depth (mm)')
        ax.legend()
        

    def summarize_gr_from_draws(Age_draws, Dep_draws, depth_grid,
                                slope_floor_yr_per_mm=0.001,
                                gr_cap_mm_per_yr=10.0):
        GR = []
        for a, d in zip(Age_draws, Dep_draws):
            # clean
            o = np.argsort(d)
            d = np.asarray(d)[o]
            a = np.maximum.accumulate(np.asarray(a)[o])

            # smooth age(depth)
            f = PchipInterpolator(d, a, extrapolate=False)
            ag = f(depth_grid)

            # skip if extrapolate created nans
            if np.any(~np.isfinite(ag)):
                continue

            gr = growth_rate_from_age_depth(
                depth_grid, ag,
                slope_floor_yr_per_mm=slope_floor_yr_per_mm,
                gr_cap_mm_per_yr=gr_cap_mm_per_yr,
                smooth_log_gr=False
            )
            GR.append(gr)

        GR = np.asarray(GR)
        if GR.size == 0:
            raise RuntimeError("No valid GR draws after cleaning.")

        # log-space percentiles (robust)
        eps = 1e-12
        logG = np.log(np.clip(GR, eps, np.inf))
        lo = np.exp(np.percentile(logG, 2.5, axis=0))
        md = np.exp(np.percentile(logG, 50, axis=0))
        hi = np.exp(np.percentile(logG, 97.5, axis=0))

        return lo, md, hi

    def Build_Summary_AgeDepth_Growth(
        self,
        depth_grid=None,
        n_grid=400,
        slope_floor_yr_per_mm=1.0,  # e.g., GR_max=1 mm/yr => 1 yr/mm
        gr_cap_mm_per_yr=None,      # e.g., 2.0
        smooth_log_gr=True,
        gr_savgol_win=31,
        gr_savgol_poly=2,
        qs=(2.5, 50, 97.5),
    ):
        """
        Returns a tidy DataFrame with Depth, Age (median/95%), and Growth Rate (median/95%).
        Growth rate is computed per posterior draw on a common depth grid using:
          - monotone enforcement
          - PCHIP smoothing
          - slope floor (prevents 1e16 blowups)
          - optional GR cap
        GR intervals summarized in log space.
        """

        # ---- collect draws across chains ----
        self.Ensure_Chain_Results()
        Age_all = []
        Dep_all = []
        for (Age_store, Dep_store, Gr_store, post_store) in self.Chain_Results:
            if Age_store is None or Age_store.shape[0] == 0:
                continue
            Age_all.append(np.asarray(Age_store, float))
            Dep_all.append(np.asarray(Dep_store, float))

        if len(Age_all) == 0:
            raise RuntimeError("No posterior draws found. Run Run_MCMC_Strat() first.")

        Age_all = np.vstack(Age_all)  # (M, K)
        Dep_all = np.vstack(Dep_all)  # (M, K)

        # ---- define depth grid ----
        dmin = float(np.nanmin(self.Depths))
        dmax = float(np.nanmax(self.Depths))
        if depth_grid is None:
            depth_grid = np.linspace(dmin, dmax, int(n_grid))
        depth_grid = np.asarray(depth_grid, float)

        M = Age_all.shape[0]
        G = depth_grid.size

        Age_on_grid = np.empty((M, G), float)
        GR_on_grid  = np.empty((M, G), float)
        Age_on_grid[:] = np.nan
        GR_on_grid[:]  = np.nan

        kept = 0
        for m in range(M):
            ag = _safe_pchip(Dep_all[m], Age_all[m], depth_grid)
            if ag is None:
                continue

            gr = _growth_rate_from_age_grid(
                depth_grid, ag,
                slope_floor_yr_per_mm=slope_floor_yr_per_mm,
                gr_cap_mm_per_yr=gr_cap_mm_per_yr,
                smooth_log_gr=smooth_log_gr,
                win=gr_savgol_win,
                poly=gr_savgol_poly,
            )

            Age_on_grid[kept, :] = ag
            GR_on_grid[kept, :]  = gr
            kept += 1

        if kept == 0:
            raise RuntimeError("All posterior draws failed during smoothing/interp. Check your Depth/Age draws.")

        Age_on_grid = Age_on_grid[:kept, :]
        GR_on_grid  = GR_on_grid[:kept, :]

        # ---- summarize Age in normal space ----
        qlo, qmed, qhi = qs
        Age_low  = np.percentile(Age_on_grid, qlo, axis=0)
        Age_med  = np.percentile(Age_on_grid, qmed, axis=0)
        Age_high = np.percentile(Age_on_grid, qhi, axis=0)

        # ---- summarize GR in log space (robust) ----
        GR_low, GR_med, GR_high = _logspace_percentiles(GR_on_grid, qs=qs)

        out = pd.DataFrame({
            "Depth_Median": depth_grid,
            "Age_Median": Age_med,
            "Age_low_95": Age_low,
            "Age_high_95": Age_high,
            "GR_Median": GR_med,
            "GR_low_95": GR_low,
            "GR_high_95": GR_high,
        })

        # optional: drop any remaining NaNs (should be rare)
        out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return out


        
        
        
    # ==============================================
    # ==============================================
    # ==============================================
    # ========================= EXTENDING ============
    # ==============================================
    # ==============================================


    def _boundary_slope_at(self, d0, d, a, side="bot", eps=1e-3):
        """
        Robust slope estimate at depth d0 using finite differencing on the
        *interpolated* curve. eps is fraction of span (NOT absolute).
        This avoids tiny Δdepth between model nodes causing crazy slopes.
        """
        d = np.asarray(d, float); a = np.asarray(a, float)
        d = np.maximum.accumulate(np.clip(d, 0.0, None))
        a = np.maximum.accumulate(np.clip(a, 0.0, None))
        p = np.argsort(d)
        d = d[p]; a = a[p]

        dmin, dmax = float(d[0]), float(d[-1])
        span = max(dmax - dmin, 1e-9)
        h = max(float(eps) * span, 1e-9)

        d0 = float(np.clip(d0, dmin, dmax))
        if side == "top":
            d1 = float(np.clip(d0 + h, dmin, dmax))
            a0 = float(np.interp(d0, d, a))
            a1 = float(np.interp(d1, d, a))
            return max(0.0, (a1 - a0) / max(d1 - d0, 1e-12))
        else:
            d1 = float(np.clip(d0 - h, dmin, dmax))
            a0 = float(np.interp(d0, d, a))
            a1 = float(np.interp(d1, d, a))
            return max(0.0, (a0 - a1) / max(d0 - d1, 1e-12))


    def _get_all_draws(self):
        """Collect all kept posterior draws across chains."""
        self.Ensure_Chain_Results()
        Age_all = []
        Dep_all = []
        for (Age_store, Dep_store, Gr_store, post_store) in self.Chain_Results:
            if Age_store is None or Age_store.shape[0] == 0:
                continue
            Age_all.append(np.asarray(Age_store, float))
            Dep_all.append(np.asarray(Dep_store, float))
        if len(Age_all) == 0:
            return np.empty((0, self.resolution)), np.empty((0, self.resolution))
        return np.vstack(Age_all), np.vstack(Dep_all)


    def _collect_boundary_slopes(self, Age_draws, Dep_draws, side="bot", kseg=12, eps=1e-12):
        """
        Collect local segment slopes near top/bottom from posterior draws.
        We compute slopes over the first/last (kseg) segments, filtering bad dd/da.
        """
        slopes = []
        M = Age_draws.shape[0]
        for m in range(M):
            a = np.maximum.accumulate(np.clip(np.asarray(Age_draws[m], float), 0.0, None))
            d = np.maximum.accumulate(np.clip(np.asarray(Dep_draws[m], float), 0.0, None))
            p = np.argsort(d)
            d = d[p]; a = a[p]

            if d.size < kseg + 2:
                continue

            if side == "bot":
                dd = np.diff(d[-(kseg+1):])
                da = np.diff(a[-(kseg+1):])
            else:
                dd = np.diff(d[:(kseg+1)])
                da = np.diff(a[:(kseg+1)])

            good = (dd > eps) & (da > 0.0)
            if np.any(good):
                s = da[good] / np.maximum(dd[good], eps)
                s = np.clip(s, eps, np.inf)
                slopes.extend(s.tolist())

        slopes = np.asarray(slopes, float)
        if slopes.size == 0:
            slopes = np.array([1e-6], float)
        return slopes


    def _ou_params_from_posterior(
        self,
        Age_draws,
        Dep_draws,
        side="bot",
        cap_lo_q=0.01,
        cap_hi_q=0.99,
        cap_expand=3.0,
        sigma_mult=1.0,
    ):
        """
        Set OU params in log-slope space from posterior boundary slopes:
          log s evolves with mean-reversion to mu_log and innovation sigma_log.
        Also returns hard slope caps [s_lo, s_hi] (expanded a bit).
        """
        slopes = self._collect_boundary_slopes(Age_draws, Dep_draws, side=side)
        slopes = np.clip(slopes, 1e-12, np.inf)
        logS = np.log(slopes)

        mu_log = float(np.median(logS))

        # caps from quantiles, expanded
        lo = float(np.quantile(slopes, cap_lo_q))
        hi = float(np.quantile(slopes, cap_hi_q))
        lo = max(lo / float(cap_expand), 1e-12)
        hi = max(hi * float(cap_expand), lo * 10.0)

        # innovation scale in log space from IQR (robust)
        iqr = float(np.quantile(logS, 0.75) - np.quantile(logS, 0.25))
        sigma_log = max(0.05, 0.5 * iqr) * float(sigma_mult)

        return mu_log, lo, hi, sigma_log


    def _ou_extrapolate_one_side(
        self,
        dg_sorted,          # sorted depths grid
        idxs,               # indices into dg_sorted to fill (either left or right side)
        start_depth,        # boundary depth (d_obs_min or d_obs_max)
        start_age,          # age at boundary
        start_slope,        # slope at boundary (>=0)
        mu_log, s_lo, s_hi, sigma_log,
        kappa,
        direction="right",  # "right" (deeper) or "left" (shallower)
        rng=None,
        min_slope=1e-12,
    ):
        """
        OU in log slope:
          d log s = kappa*(mu - log s)*dt + sigma*sqrt(dt)*N(0,1)
        and integrate age using slope s.
        """
        if rng is None:
            rng = np.random.default_rng(0)

        log_lo = float(np.log(max(s_lo, 1e-12)))
        log_hi = float(np.log(max(s_hi, max(s_lo, 1e-12) * 1.000001)))

        log_s = float(np.log(max(start_slope, min_slope)))
        log_s = float(np.clip(log_s, log_lo, log_hi))

        prev_d = float(start_depth)
        cur_y = float(start_age)

        if direction == "right":
            for j in idxs:
                dd = float(dg_sorted[j] - prev_d)
                dd = max(dd, 0.0)
                # OU update
                log_s += float(kappa) * (float(mu_log) - log_s) * dd
                log_s += float(rng.normal(0.0, sigma_log * np.sqrt(max(dd, 1e-12))))
                log_s = float(np.clip(log_s, log_lo, log_hi))
                s = max(float(np.exp(log_s)), min_slope)
                cur_y = cur_y + s * dd
                cur_y = max(0.0, cur_y)
                prev_d = float(dg_sorted[j])
                yield j, cur_y

        else:  # direction == "left"
            # walk from boundary toward smaller depth; idxs should be provided in reverse order
            for j in idxs:
                dd = float(prev_d - dg_sorted[j])
                dd = max(dd, 0.0)
                log_s += float(kappa) * (float(mu_log) - log_s) * dd
                log_s += float(rng.normal(0.0, sigma_log * np.sqrt(max(dd, 1e-12))))
                log_s = float(np.clip(log_s, log_lo, log_hi))
                s = max(float(np.exp(log_s)), min_slope)
                cur_y = cur_y - s * dd
                cur_y = max(0.0, cur_y)
                prev_d = float(dg_sorted[j])
                yield j, cur_y


    def Posterior_on_depth_grid(
        self,
        depth_grid,
        mode="ou",
        # OU controls (only used when mode="ou")
        ou_kappa=None,              # None => auto
        ou_sigma_mult=1.0,
        ou_cap_expand=3.0,
        ou_cap_lo_q=0.01,
        ou_cap_hi_q=0.99,
        # growth_rw controls (only used when mode="growth_rw")
        extrap_tau=None,
        extrap_tau_factor=1.5,
        min_slope=1e-12,
        # optional CI-only widening outside data (keeps median fixed)
        ci_inflate_kappa=0.0,
        ci_inflate_power=1.0,
        seed=0,
        qs=(2.5, 50, 97.5)
    ):
        """
        Posterior age(depth) on an arbitrary depth_grid.

        mode:
          - "model"     : require depth_grid within observed depth span
          - "clip"      : clip depths into observed range (no true extrap)
          - "linear"    : linear extrap using boundary slopes
          - "growth_rw" : random walk in log growth rate (can drift; use with caps if you must)
          - "ou"        : OU mean-reverting in log growth rate (recommended)

        Notes:
          - OU uses posterior boundary slope distribution to define caps + volatility.
          - OU extrap uncertainty grows naturally but cannot blow up to 1e97.
        """
        depth_grid = np.asarray(depth_grid, float)
        depth_grid = np.clip(depth_grid, 0.0, None)

        Age_draws, Dep_draws = self._get_all_draws()
        if Age_draws.shape[0] == 0:
            raise RuntimeError("No posterior draws found. Run Run_MCMC_Strat() first.")

        d_obs_min = float(np.min(self.Depths))
        d_obs_max = float(np.max(self.Depths))

        if mode == "model":
            if np.any(depth_grid < d_obs_min) or np.any(depth_grid > d_obs_max):
                raise ValueError("mode='model' requires depth_grid within observed depth span.")
        if mode not in ("model", "clip", "linear", "growth_rw", "ou"):
            raise ValueError("mode must be one of: 'model', 'clip', 'linear', 'growth_rw', 'ou'")

        rng = np.random.default_rng(seed)
        M = Age_draws.shape[0]
        G = depth_grid.size
        Y = np.empty((M, G), float)

        # sorted depth order for monotone enforcement
        perm = np.argsort(depth_grid)
        dg = depth_grid[perm]

        # OU parameterization (global, from posterior boundary slopes)
        if mode == "ou":
            mu_top, slo_top, shi_top, sig_top = self._ou_params_from_posterior(
                Age_draws, Dep_draws, side="top",
                cap_lo_q=ou_cap_lo_q, cap_hi_q=ou_cap_hi_q,
                cap_expand=ou_cap_expand, sigma_mult=ou_sigma_mult
            )
            mu_bot, slo_bot, shi_bot, sig_bot = self._ou_params_from_posterior(
                Age_draws, Dep_draws, side="bot",
                cap_lo_q=ou_cap_lo_q, cap_hi_q=ou_cap_hi_q,
                cap_expand=ou_cap_expand, sigma_mult=ou_sigma_mult
            )

            # kappa scale: mean reversion over ~1/3 of observed span by default
            span = max(d_obs_max - d_obs_min, 1e-6)
            if ou_kappa is None:
                kappa = 3.0 / span
            else:
                kappa = float(ou_kappa)

        # (optional) growth_rw tau inference — keep very conservative if used
        if mode == "growth_rw":
            # If you didn't define _infer_extrap_tau earlier, use a safe fallback:
            if extrap_tau is None:
                # fallback volatility (log-slope units per sqrt(mm))
                extrap_tau = 0.10 * float(extrap_tau_factor)
            else:
                extrap_tau = float(extrap_tau)

        for m in range(M):
            a = np.asarray(Age_draws[m], float)
            d = np.asarray(Dep_draws[m], float)

            d = np.maximum.accumulate(np.clip(d, 0.0, None))
            a = np.maximum.accumulate(np.clip(a, 0.0, None))
            p = np.argsort(d)
            d = d[p]; a = a[p]

            # base interpolation on clipped depths (always)
            dc = np.clip(dg, d_obs_min, d_obs_max)
            y = np.interp(dc, d, a)

            # boundary ages
            y_min = float(np.interp(d_obs_min, d, a))
            y_max = float(np.interp(d_obs_max, d, a))

            if mode == "linear":
                s_top = self._boundary_slope_at(d_obs_min, d, a, side="top")
                s_bot = self._boundary_slope_at(d_obs_max, d, a, side="bot")

                left = dg < d_obs_min
                right = dg > d_obs_max
                if np.any(left):
                    y[left] = y_min + s_top * (dg[left] - d_obs_min)
                if np.any(right):
                    y[right] = y_max + s_bot * (dg[right] - d_obs_max)

            elif mode == "growth_rw":
                # Random walk in log slope (can drift); at least cap it to avoid madness
                # caps derived from posterior boundary slopes, expanded
                slopes_bot = self._collect_boundary_slopes(Age_draws, Dep_draws, side="bot")
                slopes_top = self._collect_boundary_slopes(Age_draws, Dep_draws, side="top")
                slo = max(np.quantile(np.r_[slopes_bot, slopes_top], 0.01) / 5.0, 1e-12)
                shi = max(np.quantile(np.r_[slopes_bot, slopes_top], 0.99) * 5.0, slo * 10.0)
                log_lo = float(np.log(slo))
                log_hi = float(np.log(shi))

                # right side
                right_idx = np.where(dg > d_obs_max)[0]
                if right_idx.size > 0:
                    log_s = float(np.log(max(self._boundary_slope_at(d_obs_max, d, a, side="bot"), min_slope)))
                    log_s = float(np.clip(log_s, log_lo, log_hi))
                    prev_d = d_obs_max
                    cur_y = y_max
                    for j in right_idx:
                        dd = float(dg[j] - prev_d)
                        log_s += float(rng.normal(0.0, extrap_tau * np.sqrt(max(dd, 1e-12))))
                        log_s = float(np.clip(log_s, log_lo, log_hi))
                        s = max(float(np.exp(log_s)), min_slope)
                        cur_y = cur_y + s * dd
                        y[j] = cur_y
                        prev_d = float(dg[j])

                # left side
                left_idx = np.where(dg < d_obs_min)[0]
                if left_idx.size > 0:
                    log_s = float(np.log(max(self._boundary_slope_at(d_obs_min, d, a, side="top"), min_slope)))
                    log_s = float(np.clip(log_s, log_lo, log_hi))
                    prev_d = d_obs_min
                    cur_y = y_min
                    for j in left_idx[::-1]:
                        dd = float(prev_d - dg[j])
                        log_s += float(rng.normal(0.0, extrap_tau * np.sqrt(max(dd, 1e-12))))
                        log_s = float(np.clip(log_s, log_lo, log_hi))
                        s = max(float(np.exp(log_s)), min_slope)
                        cur_y = max(0.0, cur_y - s * dd)
                        y[j] = cur_y
                        prev_d = float(dg[j])

            elif mode == "ou":
                # OU mean-reverting log-slope with posterior caps (stable)
                # Right (deeper)
                right_idx = np.where(dg > d_obs_max)[0]
                if right_idx.size > 0:
                    s0 = max(self._boundary_slope_at(d_obs_max, d, a, side="bot"), min_slope)
                    s0 = float(np.clip(s0, slo_bot, shi_bot))
                    for j, val in self._ou_extrapolate_one_side(
                        dg_sorted=dg,
                        idxs=right_idx,
                        start_depth=d_obs_max,
                        start_age=y_max,
                        start_slope=s0,
                        mu_log=mu_bot,
                        s_lo=slo_bot,
                        s_hi=shi_bot,
                        sigma_log=sig_bot,
                        kappa=kappa,
                        direction="right",
                        rng=rng,
                        min_slope=min_slope,
                    ):
                        y[j] = val

                # Left (shallower)
                left_idx = np.where(dg < d_obs_min)[0]
                if left_idx.size > 0:
                    s0 = max(self._boundary_slope_at(d_obs_min, d, a, side="top"), min_slope)
                    s0 = float(np.clip(s0, slo_top, shi_top))
                    # walk from boundary toward smaller depth => reverse order
                    for j, val in self._ou_extrapolate_one_side(
                        dg_sorted=dg,
                        idxs=left_idx[::-1],
                        start_depth=d_obs_min,
                        start_age=y_min,
                        start_slope=s0,
                        mu_log=mu_top,
                        s_lo=slo_top,
                        s_hi=shi_top,
                        sigma_log=sig_top,
                        kappa=kappa,
                        direction="left",
                        rng=rng,
                        min_slope=min_slope,
                    ):
                        y[j] = val

            # Enforce monotone age with depth in sorted depth order
            y = np.clip(y, 0.0, None)
            y = np.maximum.accumulate(y)

            Y[m, perm] = y

        # ---- percentiles (arbitrary) ----
        qs = np.asarray(qs, float)
        P = np.percentile(Y, qs, axis=0)  # shape: (len(qs), G)

        # optional CI-only widening outside observed domain (widen outermost bands)
        if ci_inflate_kappa and ci_inflate_kappa > 0.0:
            dist = np.where(
                depth_grid < d_obs_min, d_obs_min - depth_grid,
                np.where(depth_grid > d_obs_max, depth_grid - d_obs_max, 0.0)
            )
            extra = float(ci_inflate_kappa) * (dist ** float(ci_inflate_power))
            z = 1.959963984540054
            P[0, :]  = P[0, :]  - z * extra
            P[-1, :] = P[-1, :] + z * extra

        P = np.clip(P, 0.0, None)

        # return (depth_grid, p(q1), p(q2), ...)
        return (depth_grid,) + tuple(P[i, :] for i in range(P.shape[0]))



    def Extrapolate(
        self,
        depth_low=None,
        depth_high=None,
        n=400,
        mode="ou",
        seed=0,
        **kwargs
    ):
        """
        Convenience wrapper: build a depth grid and return posterior curve on it.
        kwargs are passed to Posterior_on_depth_grid.
        """
        if depth_low is None:
            depth_low = float(np.min(self.Depths))
        if depth_high is None:
            depth_high = float(np.max(self.Depths))

        depth_low = max(0.0, float(depth_low))
        depth_high = float(depth_high)
        if depth_high <= depth_low:
            raise ValueError("depth_high must be > depth_low")

        depth_grid = np.linspace(depth_low, depth_high, int(n))
        return self.Posterior_on_depth_grid(depth_grid, mode=mode, seed=seed, **kwargs)


    def Plot_AgeDepth_Extension(
        self,
        depth_high,
        depth_low=None,
        n_base=300,
        n_ext=500,
        base_mode="clip",
        ext_mode="ou",           # <-- use "ou" here
        seed=0,
        show_ci=True,
        **kwargs
    ):
        """
        Plot standard posterior (observed range) and extended posterior on top.
        kwargs are forwarded to Posterior_on_depth_grid for the extended curve.
        """
        d_obs_min = float(np.min(self.Depths))
        d_obs_max = float(np.max(self.Depths))

        if depth_low is None:
            depth_low = d_obs_min
        depth_low = max(0.0, float(depth_low))
        depth_high = float(depth_high)
        if depth_high <= depth_low:
            raise ValueError("depth_high must be > depth_low")

        base_grid = np.linspace(d_obs_min, d_obs_max, int(n_base))
        ext_grid  = np.linspace(depth_low, depth_high, int(n_ext))

        d0, lo0, med0, hi0 = self.Posterior_on_depth_grid(
            base_grid, mode=base_mode, seed=seed, qs=(2.5, 50, 97.5)
        )

        d1, lo1, med1, hi1 = self.Posterior_on_depth_grid(
            ext_grid, mode=ext_mode, seed=seed, qs=(2.5, 50, 97.5), **kwargs
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 8))

        err_asymm = [self.low_age_err * 2.0, self.high_age_err * 2.0]
        ax.errorbar(
            x=self.u_ages, y=self.Depths, xerr=err_asymm,
            fmt="o", capsize=4, label="U–Th ages (2σ)"
        )

        ax.plot(med0, d0, label="Posterior median (observed range)")
        if show_ci:
            ax.plot(lo0, d0, ls="--", alpha=0.8, label="95% CI (observed)")
            ax.plot(hi0, d0, ls="--", alpha=0.8)

        ax.plot(med1, d1, ls=":", lw=2, label=f"Posterior median (extended: {ext_mode})")
        if show_ci:
            ax.plot(lo1, d1, ls=":", alpha=0.8, label="95% CI (extended)")
            ax.plot(hi1, d1, ls=":", alpha=0.8)

        ax.axhline(d_obs_max, ls="--", alpha=0.7)
        ax.set_xlabel("Age")
        ax.set_ylabel("Depth (mm)")
        ax.invert_yaxis()
        ax.legend(frameon=True)
        plt.tight_layout()
        return fig, ax


    def Save_Age_Depth_Model_Extended(
        self,
        depth_low=None,
        depth_high=None,
        n=400,
        mode="ou",
        seed=0,
        filename=None,
        **kwargs
    ):
        """
        Save posterior age-depth curve on an arbitrary depth grid to CSV.
        kwargs forwarded to Posterior_on_depth_grid.
        """
        if depth_low is None:
            depth_low = float(np.min(self.Depths))
        if depth_high is None:
            depth_high = float(np.max(self.Depths))

        depth_low = max(0.0, float(depth_low))
        depth_high = float(depth_high)
        if depth_high <= depth_low:
            raise ValueError("depth_high must be > depth_low")

        depth_grid = np.linspace(depth_low, depth_high, int(n))

        d, lo, med, hi = self.Posterior_on_depth_grid(
            depth_grid, mode=mode, seed=seed, **kwargs
        )

        out = pd.DataFrame({
            "Depth": d,
            "Age_Median": med,
            "Age_low_95": lo,
            "Age_high_95": hi
        })

        if filename is None:
            filename = f"{self.sample_name}_AgeDepth_EXT_{mode}.csv"
        out.to_csv(filename, index=False)
        return out



    # ======================================
    # ===========================================
    # ======================== SAVE ================
    # ===============================================
    # =============================================
    def Save_Age_Depth_Model(self):
        low_age, high_age, age_median = self.Get_Age_Model()
        low_depth, high_depth, depth_median = self.Get_Depth_Model()

        df_age_depth = pd.DataFrame({
            "Depth_Median": depth_median,
            "Depth_low_95": low_depth,
            "Depth_high_95": high_depth,
            "Age_Median": age_median,
            "Age_low_95": low_age,
            "Age_high_95": high_age
        })

        output_path = self.save_dir / f"{self.sample_name}_Age_Depth_Model.csv"
        df_age_depth.to_csv(output_path, index=False)

        print(f"Saved Age-Depth model to: {output_path}")



    # ========== Query Age ============
            
    def Query_Age_At_Depth(
        self,
        depth,
        depth_low=None,
        depth_high=None,
        n=2000,
        mode="ou",
        seed=0,
        return_full=False,
        **kwargs
    ):
        """
        Exact asymmetric 68% CI from posterior percentiles:
          low68 = p16
          median = p50
          high68 = p84
          1σ guess = 0.5*(p84 - p16)
        """
        depth = float(depth)
        if depth < 0:
            raise ValueError("depth must be >= 0")

        d_obs_min = float(np.min(self.Depths))
        d_obs_max = float(np.max(self.Depths))

        if depth_low is None:
            depth_low = min(d_obs_min, depth)
        if depth_high is None:
            depth_high = max(d_obs_max, depth)

        depth_low = max(0.0, float(depth_low))
        depth_high = float(depth_high)
        if depth_high <= depth_low:
            raise ValueError("depth_high must be > depth_low")

        grid = np.linspace(depth_low, depth_high, int(n))

        # request exact 16/50/84 percentiles
        d, p16, p50, p84 = self.Posterior_on_depth_grid(
            grid,
            mode=mode,
            seed=seed,
            qs=(16.0, 50.0, 84.0),
            **kwargs
        )

        age_med   = float(np.interp(depth, d, p50))
        age_low68 = float(np.interp(depth, d, p16))
        age_high68= float(np.interp(depth, d, p84))

        age_1s = 0.5 * (age_high68 - age_low68)

        if return_full:
            return dict(
                depth=depth,
                age_median=age_med,
                age_low68=age_low68,
                age_high68=age_high68,
                age_1sigma=age_1s,
                depth_grid=d,
                p16_grid=p16,
                p50_grid=p50,
                p84_grid=p84,
            )

        return age_med, age_low68, age_high68, age_1s
