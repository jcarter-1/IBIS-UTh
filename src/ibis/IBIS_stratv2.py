import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.interpolate import interp1d
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
        accept = (log_post_prop > log_post_cur) or (np.random.rand() <  np.exp(log_post_prop - log_post_cur))
    
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
        thin = int(self.thin)

        # how many kept?
        if niters <= 1 or niters - 1 <= burn:
            n_keep = 0
        else:
            n_keep = 1 + ((niters - 1 - burn) // thin)

        # allocate kept draws
        Age_store = np.empty((n_keep, Nmodel)) if n_keep > 0 else np.empty((0, Nmodel))
        Dep_store = np.empty((n_keep, Nmodel)) if n_keep > 0 else np.empty((0, Nmodel))
        Gr_store = np.empty((n_keep, Nmodel - 1)) if n_keep > 0 else np.empty((0, Nmodel - 1))
        post_store = np.empty((n_keep,)) if n_keep > 0 else np.empty((0,))

        # acceptance accounting
        proposal_counts = {k: 0 for k in tuning_factors}
        accept_counts = {k: 0 for k in tuning_factors}
        total_props = 0
        total_accs = 0

        # checkpoint start
        with open(self._theta_path(chain_id), 'wb') as f:
            pickle.dump(theta, f)
        with open(self._tuning_path(chain_id), 'wb') as f:
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



            # keep thinned post-burn draws
            if (i >= burn) and (((i - burn) % thin) == 0) and (save_idx < n_keep):
                Age_store[save_idx, :] = Age_Model
                Dep_store[save_idx, :] = Depth_Model
                Gr_store[save_idx, :] = np.diff(Age_Model) / np.maximum(np.diff(Depth_Model), 1e-12)
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
                with open(self._theta_path(chain_id), 'wb') as f:
                    pickle.dump((Age_Model, Depth_Model), f)
                with open(self._tuning_path(chain_id), 'wb') as f:
                    pickle.dump(tuning_factors, f)

        # final checkpoint
        with open(self._theta_path(chain_id), 'wb') as f:
            pickle.dump((Age_Model, Depth_Model), f)
        with open(self._tuning_path(chain_id), 'wb') as f:
            pickle.dump(tuning_factors, f)

        return Age_store, Dep_store, Gr_store, post_store


    # ---------- SEED Everything ----------
    def check_starting_parameters(self):
        thetas = []
        for chain_id in range(self.n_chains):
            path = self._theta_path(chain_id)
            if self.Start_from_pickles and os.path.exists(path):
                with open(path, 'rb') as f:
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
        with open(self._meta_path(), 'wb') as f:
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
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    # =====================================================================================
    
    
    # =====================================================================================
    # =====================================================================================
    # ==================================== SAVE =========================================
    # =====================================================================================
    # =====================================================================================
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
