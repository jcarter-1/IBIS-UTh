import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm, uniform
from scipy.special import log_ndtr
from scipy.interpolate import interp1d
from tqdm import tqdm
import dill as pickle
import time, os, random, warnings
from joblib import Parallel, delayed
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ---------- fast----------
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
def _norm_logpdf_vector(x, mu, sig):
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sig = np.asarray(sig, float)

    sig = np.maximum(sig, 1e-12)  # avoid division by zero

    z = (x - mu) / sig
    return -0.5 * z**2 - np.log(sig) - _LOG_SQRT_2PI

class ThoriumPrior:
    def __init__(self, thor_kde, thor_min, thor_max, rng=None):
        self.Thor_KDE = thor_kde
        self.thor_xmax = self._estimate_thor_xmax()
        grid = np.linspace(0.0, self.thor_xmax, 1024)
        self.grid = np.asarray(grid, float)
        self.Thor_KDE = thor_kde
        self.TH0_MIN = float(thor_min)
        self.TH0_MAX = float(thor_max)
        self.rng = rng if rng is not None else np.random.default_rng()

        if self.grid.ndim != 1 or self.grid.size < 2:
            raise ValueError("grid must be a 1D array with at least 2 points")
        if np.any(~np.isfinite(self.grid)):
            raise ValueError("grid contains non-finite values")
        if np.any(np.diff(self.grid) <= 0):
            raise ValueError("grid must be strictly increasing")

        self._build_thor_inv_cdf()

    def thor_pdf(self, x):
        x = np.asarray(x, float)
        return np.asarray(self.Thor_KDE(x), float)

    def _estimate_thor_xmax(self, x_min=0.0,
                            q=0.9995,
                            n_try=20000,
                            max_rounds=8):
        """
        Estimate a sensible upper support
        bound for plotting / inverse-CDF sampling.
        """
        x_hi = 1.0
        for _ in range(max_rounds):
            grid = np.linspace(x_min, x_hi, 1024)
            pdf = self.thor_pdf(grid)
            peak = np.nanmax(pdf)
            tail = pdf[-1]
            if np.isfinite(peak) and peak > 0 and tail / peak < 1e-6:
                return float(x_hi)
            x_hi *= 2.0

        return float(x_hi)
        
    def thor_logpdf(self, x):
        vals = np.asarray(self.thor_pdf(x), float)
        vals = np.clip(vals, 1e-300, None)
        return np.log(vals)

    def _build_thor_inv_cdf(self):
        x = self.grid
        pdf = self.thor_pdf(x)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf = np.clip(pdf, 0.0, None)

        # enforce support
        pdf[(x < self.TH0_MIN) | (x > self.TH0_MAX)] = 0.0

        area = np.trapz(pdf, x)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Thorium prior pdf has zero or invalid area on the supplied grid.")

        pdf = pdf / area

        dx = np.diff(x)
        cdf = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
        ])
        cdf[-1] = 1.0

        keep = np.r_[True, np.diff(cdf) > 0]
        cdf_use = cdf[keep]
        x_use = x[keep]

        if cdf_use.size < 2:
            raise ValueError("Inverse CDF construction failed: degenerate CDF.")

        self._thor_inv_cdf = interp1d(
            cdf_use,
            x_use,
            bounds_error=False,
            fill_value=(x_use[0], x_use[-1]),
            assume_sorted=True,
        )

    def rvs(self, n):
        n = int(n)
        u = self.rng.random(n)
        out = np.asarray(self._thor_inv_cdf(u), float)
        return np.clip(out, self.TH0_MIN, self.TH0_MAX)

class IBIS_MCMC:
    """
    IBIS MCMC for per-sample initial Th with strat constraints.

    Speed-ups (math unchanged):
      - Cache ln_prior as sum of per-index terms; O(1) update for single-index moves
      - Cache strat likelihood sum over all pairs; O(N) update for single-index moves
      - Keep full evaluation for block moves / multi-index changes

    IMPORTANT:
      - Cache rollback is handled safely (no stale rollback pack reuse).
    """

    def __init__(self, Thor_KDE, Age_Maximum,
                 Age_Uncertainties, data, sample_name='SAMPLE_NAME',
                 n_chains=3, iterations=50000, burn_in=10000,
                 Start_from_pickles=True, method='thoth',
                 save_dir = None):

        self.method = method
        self.data = data
        self.burn_in = int(burn_in)
        self.Age_Maximum = float(Age_Maximum)
        self.Thor_KDE = Thor_KDE
        self.rng = np.random.default_rng()
        # depths + order (shallow->deep)
        self.depths = np.asarray(self.data['Depths'].values, float)
        self._depth_order = np.argsort(self.depths)
        self.depths = self.depths[self._depth_order]

        self.Age_Solve_Max = 5.0 * float(self.Age_Maximum)
        self.Age_Uncertainties = np.asarray(Age_Uncertainties, float)[self._depth_order]

        self.Th230_lam = 9.17055e-06   # Cheng et al. (2013)
        self.Th230_lam_err = 6.67e-09
        self.U234_lam = 2.82203e-06
        self.U234_lam_err = 1.494e-09

        self.N_meas = int(data.shape[0])
        self.n_chains = int(n_chains)
        self.iterations = int(iterations)
        self.sample_name = sample_name
        self.Chain_Results = None
        self.Start_from_pickles = bool(Start_from_pickles)
        
        # Create a save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()
        
        self.save_dir.mkdir(parents = True, exist_ok = True)

        self.Depths = self.data['Depths'].values
        self.Depths_err = self.data['Depths_err'].values

        # measured ratios in depth order
        self.r08 = np.asarray(data['Th230_238U_ratios'].values[self._depth_order], float)
        self.r28 = np.asarray(data['Th232_238U_ratios'].values[self._depth_order], float)
        self.r48 = np.asarray(data['U234_U238_ratios'].values[self._depth_order], float)
        self.r08_err = np.asarray(data['Th230_238U_ratios_err'].values[self._depth_order], float)
        self.r28_err = np.asarray(data['Th232_238U_ratios_err'].values[self._depth_order], float)
        self.r48_err = np.asarray(data['U234_U238_ratios_err'].values[self._depth_order], float)
        # bounds for Th0
        self.TH0_MIN = 0.0001
        self.TH0_MAX = self.compute_TH0_MAX(self.r08, self.r08_err, self.r28, self.r28_err)
        
        # Get Thorium Prior Object here
        self.thorium_prior = ThoriumPrior(
            thor_kde=Thor_KDE,
            thor_min=self.TH0_MIN,
            thor_max=self.TH0_MAX,
            rng= self.rng,
        )
    
        def _safe_log_step(val, err, floor=1e-8):
            val = max(float(val), 1e-12)
            err = max(float(err), floor)
            rel = err / val
            return float(np.sqrt(np.log(1.0 + rel**2)))
        
        # --- move probabilities ---
        self.p_block  = 0.65
        self.p_single = 0.35
        
        # --- block tuning: one scale per sample index ---
        self.tuning_block = {}
        for i in range(self.N_meas):
            self.tuning_block[f'Block_{i}'] = 0.10
        
        # --- single-parameter tuning: one scale per parameter per index ---
        self.tuning_single = {}
        for i in range(self.N_meas):
            self.tuning_single[f'Initial_Thorium_{i}']   = 0.10
            self.tuning_single[f'U234_U238_ratios_{i}']  = _safe_log_step(self.r48[i], self.r48_err[i])
            self.tuning_single[f'Th230_U238_ratios_{i}'] = _safe_log_step(self.r08[i], self.r08_err[i])
            self.tuning_single[f'Th232_U238_ratios_{i}'] = _safe_log_step(self.r28[i], self.r28_err[i])
        
        self.block_keys  = list(self.tuning_block.keys())
        self.single_keys = list(self.tuning_single.keys())
        self.keys = self.block_keys + self.single_keys
    
        # Pairings
        N = self.N_meas
        I, J = np.triu_indices(N, k=1)
        self._IJ_I = I
        self._IJ_J = J

        errs = np.asarray(self.Age_Uncertainties, float)
        self._pair_sigma = np.hypot(errs[I], errs[J])

        d = self.depths
        self._pair_same = (d[J] == d[I])
        self._pair_diff = ~self._pair_same

        # pairs that touch each index (for O(N) incremental updates)
        self._pairs_by_idx = [None] * N
        for k in range(N):
            self._pairs_by_idx[k] = np.where((I == k) | (J == k))[0]

        # adjacent pairs (for hard/hinge checks)
        self._depth_order = np.argsort(self.depths)
        self._adj_I = self._depth_order[:-1]
        self._adj_J = self._depth_order[1:]
        self._adj_sigma = np.hypot(errs[self._adj_I], errs[self._adj_J])
        
        # storage
        self.store_thin = 5
        self.store_dtype = np.float32
        
        # moves
        self._move_funcs = [
            ("PerSampleBlock", self.PerSampleBlock_Move),
        ]

        # caches
        self._ages_cache = None
        self._lp_u = None
        self._lp_k = None
        self._lp_r = None
        self._lp_prior_total = None
        self._ll_strat_total = None

        
    # ---------------- bounds helper ----------------
    def compute_TH0_MAX(self, r08, r08e, r28, r28e, k=3.0, eps=1e-12, l230=9.17055e-06):
        r08_hi = np.maximum(r08 + k * r08e, eps)
        r28_lo = np.maximum(r28 - k * r28e, eps)
        ok = np.isfinite(r08_hi) & np.isfinite(r28_lo) & (r28_lo > 0)
        if not np.any(ok):
            return 1e6
        r_meas = r08_hi[ok] / r28_lo[ok]
        r0_upper = r_meas * np.exp(l230 * self.Age_Solve_Max)
        hi = float(np.nanmax(r0_upper))
        hi = max(hi * 1.25, 10.0)
        return min(hi, 1e6)
        
    # ---------------- strat constraints ----------------
    def _hard_strat_check_adj(self, ages, dt_min=None):
        ages = np.asarray(ages, float)
        if np.any(~np.isfinite(ages)):
            return False
        i = self._adj_I
        j = self._adj_J
        diff_depth = (self.depths[j] != self.depths[i])
        if not np.any(diff_depth):
            return True
        if dt_min is None:
            dt_min = 1e-12 * float(self.Age_Maximum)
        delta = ages[j] - ages[i]
        return np.all(delta[diff_depth] >= dt_min)

    def _worst_adj_violation(self, ages):
        i = self._adj_I
        j = self._adj_J
        dA = ages[j] - ages[i]
        z = dA / np.maximum(self._adj_sigma, 1e-12)
        k = np.argmin(z)
        if z[k] < 0.0:
            return int(i[k]), int(j[k]), float(dA[k]), float(z[k])
        return None, None, None, None

    def _desired_th_sign(self, i, j, idx):
        return -1 if idx == j else +1

    # ---------------- age solver ----------------
    def U_series_age_equation(self, age, Th_initial, Th232_ratio, U234_ratio, Th230_ratio):
        dlam = self.Th230_lam - self.U234_lam
        L1 = Th230_ratio
        L2 = Th232_ratio * Th_initial * np.exp(-self.Th230_lam * age)
        R1 = 1.0 - np.exp(-self.Th230_lam * age)
        if abs(dlam) < 1e-12:
            return np.nan
        R2 = (U234_ratio - 1.0) * (self.Th230_lam / dlam) * (1.0 - np.exp(-dlam * age))
        return L1 - L2 - R1 - R2

    def _age_fun_and_deriv(self, a, Th_initial, Th232, U234, Th230):
        l230 = self.Th230_lam
        l234 = self.U234_lam
        dlam = l230 - l234
        e230 = np.exp(-l230 * a)
        ed   = np.exp(-dlam * a)
        f = (Th230 - Th232 * Th_initial * e230 - (1.0 - e230)
             - (U234 - 1.0) * (l230 / dlam) * (1.0 - ed))
        fp = l230 * e230 * (Th232 * Th_initial - 1.0) - l230 * (U234 - 1.0) * ed
        return f, fp

    def _solve_age_single(self, Th_initial, Th232, U234, Th230, a0, amax, newton_max=8):
        amin = 0.0
        amax = float(amax)

        def g(t):
            return self.U_series_age_equation(t, Th_initial, Th232, U234, Th230)

        # ---------- 0) quick near-zero handling ----------
        g0 = g(0.0)
        if np.isfinite(g0) and abs(g0) < 1e-50:
            return 0.0

        # Try an extremely small positive time to catch a root ~0
        tiny = min(1e-8 * amax, 1e-6)  # absolute cap so it works even if amax huge
        gt = g(tiny)
        if np.isfinite(g0) and np.isfinite(gt) and (g0 * gt <= 0.0):
            try:
                root = brentq(g, 0.0, tiny, maxiter=200)
                if np.isfinite(root) and (0.0 <= root <= amax):
                    return float(root)
            except Exception:
                pass

        # ---------- 1) Newton phase (only if a0 is usable) ----------
        a = np.clip(a0 if np.isfinite(a0) else 0.5 * amax, max(1e-12, amin), amax)

        for _ in range(int(newton_max)):
            f, fp = self._age_fun_and_deriv(a, Th_initial, Th232, U234, Th230)
            if (not np.isfinite(f)) or (not np.isfinite(fp)) or abs(fp) < 1e-14:
                break

            step = f / fp
            a_new = a - step

            if (a_new < amin) or (a_new > amax):
                a_new = np.clip(a - 0.5 * step, amin, amax)

            if abs(a_new - a) < 1e-10 * max(1.0, a):
                if np.isfinite(a_new) and (amin <= a_new <= amax):
                    return float(a_new)
                break

            a = a_new

        # ---------- 2) local bracket around Newton landing ----------
        left  = max(amin, a - 0.2 * amax)
        right = min(amax, a + 0.2 * amax)
        fL, fR = g(left), g(right)

        if np.isfinite(fL) and np.isfinite(fR) and (fL * fR <= 0.0):
            try:
                root = brentq(g, left, right, maxiter=200)
                if np.isfinite(root) and (amin <= root <= amax):
                    return float(root)
            except Exception:
                pass

        # ---------- 3) GLOBAL bracket search (log-spaced, resolves youngest root) ----------
        # Log grid has tons of resolution near 0 and still covers to amax.
        # Include 0 explicitly.
        grid = np.r_[0.0, np.logspace(-12, np.log10(amax), 600)]
        vals = np.array([g(t) for t in grid], dtype=float)

        ok = np.isfinite(vals)
        grid2 = grid[ok]
        vals2 = vals[ok]
        if vals2.size >= 2:
            s = np.sign(vals2)
            crossings = np.where(s[:-1] * s[1:] <= 0.0)[0]
            if crossings.size > 0:
                k = int(crossings[0])
                aL = float(grid2[k])
                aR = float(grid2[k + 1])
                # ensure non-degenerate interval
                if aR == aL:
                    aR = min(amax, aL + max(1e-12, 1e-12 * amax))
                try:
                    root = brentq(g, aL, aR, maxiter=200)
                    if np.isfinite(root) and (amin <= root <= amax):
                        return float(root)
                except Exception:
                    pass

        return np.nan
        
    def ages_vector(self, Th_initial, U234, Th230, Th232, age_guess=None):
        N = len(Th_initial)
        ages = np.empty(N, dtype=float)
        if age_guess is None or np.isscalar(age_guess):
            guess = np.full(N, 0.5 * self.Age_Maximum, float)
        else:
            guess = np.asarray(age_guess, float)
            if guess.shape[0] != N:
                raise ValueError("age_guess must be length N")
        amax = float(self.Age_Solve_Max)
        for i in range(N):
            ages[i] = self._solve_age_single(Th_initial[i], Th232[i], U234[i], Th230[i],
                                             a0=guess[i], amax=amax)
        return ages


    def ln_prior_ratios(self, U234, Th230, Th232):
        if np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0):
            return -np.inf
        s48 = np.clip(self.r48_err, 1e-12, np.inf)
        s08 = np.clip(self.r08_err, 1e-12, np.inf)
        s28 = np.clip(self.r28_err, 1e-12, np.inf)
        lp = 0.0
        lp += np.sum(_norm_logpdf_vector(U234, self.r48, s48))
        lp += np.sum(_norm_logpdf_vector(Th230, self.r08, s08))
        lp += np.sum(_norm_logpdf_vector(Th232, self.r28, s28))
        return float(lp)

    def _theta_is_finite_positive(self, theta):
        if theta is None or (not isinstance(theta, (tuple, list))) or len(theta) != 4:
            return False
        th, U234, Th230, Th232 = theta
        th   = np.asarray(th, float)
        U234 = np.asarray(U234, float)
        Th230= np.asarray(Th230, float)
        Th232= np.asarray(Th232, float)

        if (th.shape[0] != self.N_meas or U234.shape[0] != self.N_meas or
            Th230.shape[0] != self.N_meas or Th232.shape[0] != self.N_meas):
            return False

        if (np.any(~np.isfinite(th)) or np.any(~np.isfinite(U234)) or
            np.any(~np.isfinite(Th230)) or np.any(~np.isfinite(Th232))):
            return False

        if (np.any(th <= 0) or np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0)):
            return False

        return True

    def ln_prior(self, theta):
        if not self._theta_is_finite_positive(theta):
            return -np.inf

        th, U234, Th230, Th232 = theta
        th = np.asarray(th, float)
        U234 = np.asarray(U234, float)
        Th230 = np.asarray(Th230, float)
        Th232 = np.asarray(Th232, float)

        lp = 0.0
        # Total prior is the combination of the
        # measured ratios and initial ratio
        lp += float(np.sum(self.thorium_prior.thor_logpdf(th)))
        lp += float(self.ln_prior_ratios(U234, Th230, Th232))
        return lp if np.isfinite(lp) else -np.inf

    # ---------------- likelihood pieces ----------------
    def strat_likelihood(self, ages):
        I = self._IJ_I; J = self._IJ_J
        dA = ages[J] - ages[I]
        sig = np.maximum(self._pair_sigma, 1e-12)
        same = self._pair_same
        diff = self._pair_diff
        ll_same = norm.logpdf(dA[same], loc=0.0, scale=sig[same]).sum()
        z = dA[diff] / sig[diff]
        ll_strat = log_ndtr(z).sum()
        return float(ll_same + ll_strat)

    def age_max_penalty(self, ages, w=1e6, power=2):
        ages = np.asarray(ages, float)
        Amax = float(self.Age_Maximum)
        over = np.maximum(0.0, ages - Amax)
        denom = max(Amax, 1.0)
        x = over / denom
        return -float(w) * float(np.sum(x ** power))

    def strat_hinge_penalty_adj(self, ages, k=50.0):
        i = self._adj_I; j = self._adj_J
        diff_depth = (self.depths[j] != self.depths[i])
        if not np.any(diff_depth):
            return 0.0
        dA = ages[j] - ages[i]
        sig = np.maximum(self._adj_sigma, 1e-12)
        z = dA / sig
        neg = np.minimum(0.0, z)
        return -float(k) * float(np.sum(neg[diff_depth] ** 2))

    def _age_is_saturated(self, ages, frac=1e-6):
        eps = float(frac) * float(self.Age_Maximum)
        return np.asarray(ages, float) >= (float(self.Age_Maximum) - eps)

    # ---------------- full posterior evaluators ----------------
    def log_posterior(self, theta, ages_prev=None):
        if not self._theta_is_finite_positive(theta):
            return -np.inf, np.full(self.N_meas, np.nan)

        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf, np.full(self.N_meas, np.nan)

        th, U234, Th230, Th232 = theta
        try:
            ages = self.ages_vector(th, U234, Th230, Th232, age_guess=ages_prev)
        except RuntimeError:
            # no root / solver failure => impossible state
            return -np.inf, np.full(self.N_meas, np.nan)
        lp += self.age_max_penalty(ages, w=1e5, power=2)

        if (np.any(~np.isfinite(ages)) or np.any(ages < 0)):
            return -np.inf, ages

        if not self._hard_strat_check_adj(ages, dt_min=None):
            lp += self.strat_hinge_penalty_adj(ages, k=50)

        ll = float(self.strat_likelihood(ages))
        if not np.isfinite(ll):
            return -np.inf, ages

        sat = self._age_is_saturated(ages, frac=1e-7)
        if np.any(sat):
            lp += -10.0 * float(np.sum(sat))

        return float(lp + ll), ages

  # ---------------- proposal helpers ----------------
    def _draw_positive_normal(self, mu, sigma, max_tries=50):
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        x = self.rng.normal(mu, sigma)
        bad = x <= 0
        tries = 0
        while np.any(bad) and tries < max_tries:
            x[bad] = self.rng.normal(mu[bad], sigma[bad])
            bad = x <= 0
            tries += 1
        x = np.where(x <= 0, 1e-12, x)
        return x

    def Initial_Guesses_for_Model(self, max_attempts=20000, verbose=True):
        initial_thetas = []

        # USE DEPTH-SORTED MEANS + ERRORS (critical)
        Th230_c = self.r08
        Th232_c = self.r28
        U234_c  = self.r48
        Th230_s = self.r08_err
        Th232_s = self.r28_err
        U234_s  = self.r48_err

        # monotone age guess by depth to help Newton
        age_guess = np.linspace(0.05, 0.95, self.N_meas) * float(self.Age_Maximum)
        found = np.zeros(self.n_chains, dtype=bool)

        for chain in range(self.n_chains):
            for _ in range(int(max_attempts)):
                Th_initial = self.thorium_prior.rvs(self.N_meas)
                Th_initial = np.clip(Th_initial, self.TH0_MIN, self.TH0_MAX)

                Th230_in = self._draw_positive_normal(Th230_c, Th230_s)
                Th232_in = self._draw_positive_normal(Th232_c, Th232_s)
                U234_in  = self._draw_positive_normal(U234_c,  U234_s)

                theta = (Th_initial, U234_in, Th230_in, Th232_in)
                logp, ages = self.log_posterior(theta, ages_prev=age_guess)

                if np.isfinite(logp) and np.isfinite(ages).all():
                    initial_thetas.append(theta)
                    found[chain] = True
                    break
            if (not found[chain]) and verbose:
                print(f"Chain {chain} failed after {max_attempts} attempts")
        if not np.all(found):
            bad = np.where(~found)[0].tolist()
            raise RuntimeError(
                f"No valid starting θ found for chains {bad}. "
                f"Likely: Age_Maximum too tight / Th0 bounds too restrictive / "
                f"data imply no root for some points."
            )
        return initial_thetas, found

    # ---------------- moves ----------------
    def PerSampleBlock_Move(self, theta, index):
        th_cur, U234_cur, Th230_cur, Th232_cur = theta
    
        s_block = float(self.tuning_block.get(f'Block_{index}', 0.10))
    
        # relative weights inside the block
        w_th   = 1.00
        w_u234 = 0.60
        w_t30  = 0.60
        w_t32  = 0.60
    
        th_new    = th_cur.copy()
        U234_new  = U234_cur.copy()
        Th230_new = Th230_cur.copy()
        Th232_new = Th232_cur.copy()
    
        th_new[index]    = np.exp(np.log(max(th_cur[index],    1e-12)) + self.rng.normal(0.0, w_th   * s_block))
        U234_new[index]  = np.exp(np.log(max(U234_cur[index],  1e-12)) + self.rng.normal(0.0, w_u234 * s_block))
        Th230_new[index] = np.exp(np.log(max(Th230_cur[index], 1e-12)) + self.rng.normal(0.0, w_t30  * s_block))
        Th232_new[index] = np.exp(np.log(max(Th232_cur[index], 1e-12)) + self.rng.normal(0.0, w_t32  * s_block))
    
        th_new[index] = float(np.clip(th_new[index], self.TH0_MIN, self.TH0_MAX))
    
        return (th_new, U234_new, Th230_new, Th232_new)

    def SingleParameter_Move(self, theta, index, param_name=None):
        th_cur, U234_cur, Th230_cur, Th232_cur = theta
    
        if param_name is None:
            param_name = self.rng.choice([
                'Initial_Thorium',
                'U234_U238_ratios',
                'Th230_U238_ratios',
                'Th232_U238_ratios',
            ])
    
        th_new    = th_cur.copy()
        U234_new  = U234_cur.copy()
        Th230_new = Th230_cur.copy()
        Th232_new = Th232_cur.copy()
    
        key = f'{param_name}_{index}'
        s = float(self.tuning_single.get(key, 0.05))
    
        if param_name == 'Initial_Thorium':
            th_new[index] = np.exp(np.log(max(th_cur[index], 1e-12)) + self.rng.normal(0.0, s))
            th_new[index] = float(np.clip(th_new[index], self.TH0_MIN, self.TH0_MAX))
    
        elif param_name == 'U234_U238_ratios':
            U234_new[index] = np.exp(np.log(max(U234_cur[index], 1e-12)) + self.rng.normal(0.0, s))
    
        elif param_name == 'Th230_U238_ratios':
            Th230_new[index] = np.exp(np.log(max(Th230_cur[index], 1e-12)) + self.rng.normal(0.0, s))
    
        elif param_name == 'Th232_U238_ratios':
            Th232_new[index] = np.exp(np.log(max(Th232_cur[index], 1e-12)) + self.rng.normal(0.0, s))
    
        else:
            raise ValueError(f"Unknown parameter name: {param_name}")
    
        return (th_new, U234_new, Th230_new, Th232_new), key


    # ---------------- Propose State -----------------
    def propose_state(self, theta, index):
        r = self.rng.random()
    
        if r < self.p_block:
            theta_prop = self.PerSampleBlock_Move(theta, index)
            move_type = "block"
            moved_keys = [f'Block_{index}']
            return theta_prop, move_type, moved_keys
    
        else:
            theta_prop, single_key = self.SingleParameter_Move(theta, index)
            move_type = "single"
            moved_keys = [single_key]
            return theta_prop, move_type, moved_keys
    

    # ---------------- tuning adaptation ----------------
    def adapt_one(self, key, rate, t, target=0.25, smin=1e-5, smax=1.0):
        eta = 1.0 / np.sqrt(max(1, t))
    
        if key in self.tuning_block:
            log_s = np.log(max(float(self.tuning_block[key]), 1e-12))
            log_s += eta * (rate - target)
            self.tuning_block[key] = float(np.clip(np.exp(log_s), smin, smax))
    
        elif key in self.tuning_single:
            log_s = np.log(max(float(self.tuning_single[key]), 1e-12))
            log_s += eta * (rate - target)
            self.tuning_single[key] = float(np.clip(np.exp(log_s), smin, smax))
    
        else:
            raise KeyError(f"Unknown tuning key: {key}")

    # ---------------- I/O ----------------
    def Save_Parameters_and_Tuning(self, theta, chain_id):
        tf_file = self.save_dir / f'tuning_{self.sample_name}_{chain_id}.pkl'
        theta_file = self.save_dir / f'{self.sample_name}_theta_{chain_id}.pkl'
    
        with open(theta_file, 'wb') as f:
            pickle.dump(theta, f)
    
        tuning_payload = {
            "tuning_block": self.tuning_block,
            "tuning_single": self.tuning_single,
        }
        with open(tf_file, 'wb') as f:
            pickle.dump(tuning_payload, f)
    
    # ===================== MAIN MCMC FUNCTION =====================
    def MCMC(self, theta, iterations, chain_id):
        start_time = time.time()
        Ndata = self.N_meas
        total_iterations = int(iterations) + int(self.burn_in)

        # load per-chain tuning
        tf_file = self.save_dir / f'tuning_{self.sample_name}_{chain_id}.pkl'
        if tf_file.exists() and self.Start_from_pickles:
            with open(tf_file, 'rb') as f:
                payload = pickle.load(f)
        
            if isinstance(payload, dict) and ("tuning_block" in payload) and ("tuning_single" in payload):
                self.tuning_block = payload["tuning_block"]
                self.tuning_single = payload["tuning_single"]
            else:
                # backward compatibility with old tuning pickle
                pass

        # counters
        all_keys = self.block_keys + self.single_keys
        self.proposal_counts = {k: 0 for k in all_keys}
        self.accept_counts   = {k: 0 for k in all_keys}
        
        self._last_prop_counts = {k: 0 for k in all_keys}
        self._last_acc_counts  = {k: 0 for k in all_keys}

        self.diag = {
            "accepted": 0,
            "invalid_theta": 0,
            "age_nan": 0,
            "lp_nan": 0,
            "mh_reject": 0,
        }

        self.rej_by_index = defaultdict(lambda: {
            "accepted": 0,
            "invalid_theta": 0,
            "age_nan": 0,
            "lp_nan": 0,
            "mh_reject": 0,
        })

        if not hasattr(self, "_adapt_step"):
            self._adapt_step = 0

        # initial eval
        init_ages_guess = np.linspace(0.05, 0.95, Ndata) * float(self.Age_Maximum)
        logp_cur, ages_cur = self.log_posterior(theta, ages_prev=init_ages_guess)

        if (not np.isfinite(logp_cur)) or np.any(~np.isfinite(ages_cur)):
            raise RuntimeError(
                f"Initial theta produced non-finite posterior/ages for chain {chain_id}. "
                f"logp={logp_cur}, finite_ages={np.isfinite(ages_cur).all()}"
            )

        # storage
        keep = iterations // self.store_thin + int(iterations % self.store_thin != 0)
        Ages_store            = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Initial_Th_mean_store = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_initial_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_ratios_store     = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th232_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th230_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        posterior_store       = np.zeros(keep, dtype=self.store_dtype)

        sample_index = 0

        pbar = tqdm(
            range(1, total_iterations + 1),
            desc=f"Chain {chain_id}",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            disable=(chain_id != 0),
        )

        def _register(idx, outcome):
            self.diag[outcome] += 1
            self.rej_by_index[idx][outcome] += 1
            
        def _adapt_window(target=0.30, min_props_block=20, min_props_single=3):
            self._adapt_step += 1
            t = self._adapt_step
        
            for key in self.block_keys:
                p_tot = self.proposal_counts.get(key, 0)
                a_tot = self.accept_counts.get(key, 0)
                p_last = self._last_prop_counts.get(key, 0)
                a_last = self._last_acc_counts.get(key, 0)
        
                dp = p_tot - p_last
                da = a_tot - a_last
        
                if dp >= min_props_block:
                    rate = da / dp
                    self.adapt_one(key, rate, t, target=target, smin=1e-5, smax=1.0)
        
                self._last_prop_counts[key] = p_tot
                self._last_acc_counts[key]  = a_tot
        
            for key in self.single_keys:
                p_tot = self.proposal_counts.get(key, 0)
                a_tot = self.accept_counts.get(key, 0)
                p_last = self._last_prop_counts.get(key, 0)
                a_last = self._last_acc_counts.get(key, 0)
        
                dp = p_tot - p_last
                da = a_tot - a_last
        
                if dp >= min_props_single:
                    rate = da / dp
                    self.adapt_one(key, rate, t, target=target, smin=1e-5, smax=1.0)
        
                self._last_prop_counts[key] = p_tot
                self._last_acc_counts[key]  = a_tot

        def _print_diag(i):
            props = sum(self.proposal_counts.values()) or 1
            accs  = sum(self.accept_counts.values())
            acc_rate = accs / props

            ii, jj, dA, zz = self._worst_adj_violation(ages_cur)
            worst = "none" if ii is None else f"i={ii}, j={jj}, Δ={dA:.6g}, z={zz:.6g}"
            sat_frac = float(np.mean(self._age_is_saturated(ages_cur, frac=1e-7)))

            print(
                f"\n================ MCMC diagnostics @ iter {i} ================\n"
                f"overall acc          : {acc_rate:.3f}\n"
                f"accepted             : {self.diag['accepted']}\n"
                f"invalid_theta        : {self.diag['invalid_theta']}\n"
                f"age_nan              : {self.diag['age_nan']}\n"
                f"lp_nan               : {self.diag['lp_nan']}\n"
                f"mh_reject            : {self.diag['mh_reject']}\n"
                f"sat_frac             : {sat_frac:.3f}\n"
                f"worst_adj_violation  : {worst}\n"
            )

            print("\n--- block tuning snapshot ---")
            vals = list(self.tuning_block.values())
            if vals:
                print(
                    f"Block scales           median={np.median(vals):.4g}  "
                    f"min={np.min(vals):.4g}  max={np.max(vals):.4g}"
                )
            
            print("\n--- single-move tuning snapshot ---")
            for prefix in ["Initial_Thorium_", "U234_U238_ratios_", "Th230_U238_ratios_", "Th232_U238_ratios_"]:
                vals = [v for k, v in self.tuning_single.items() if k.startswith(prefix)]
                if vals:
                    print(
                        f"{prefix:20s} median={np.median(vals):.4g}  "
                        f"min={np.min(vals):.4g}  max={np.max(vals):.4g}"
                    )

            print("\n--- worst indices (top 8) ---")
            scored = []
            for idx, d in self.rej_by_index.items():
                total = sum(d.values())
                fail = d["invalid_theta"] + d["age_nan"] + d["lp_nan"] + d["mh_reject"]
                scored.append((idx, total, fail, d["accepted"]))
            scored.sort(key=lambda x: x[2], reverse=True)

            for idx, total, fail, acc in scored[:8]:
                print(f"idx={idx:3d}  total={total:6d}  fail={fail:6d}  acc={acc:6d}")

        for i in pbar:
            idx = int(self.rng.integers(0, Ndata))
            
            theta_prop, move_type, moved_keys = self.propose_state(theta, idx)
            
            for key in moved_keys:
                self.proposal_counts[key] += 1
            
            logp_prop, ages_prop = self.log_posterior(theta_prop, ages_prev=ages_cur)

            if not self._theta_is_finite_positive(theta_prop):
                _register(idx, "invalid_theta")
                accept = False
            elif np.any(~np.isfinite(ages_prop)):
                _register(idx, "age_nan")
                accept = False
            elif not np.isfinite(logp_prop):
                _register(idx, "lp_nan")
                accept = False
            else:
                accept = (np.log(self.rng.random()) < (logp_prop - logp_cur))
                if accept:
                    theta = theta_prop
                    logp_cur = float(logp_prop)
                    ages_cur = np.asarray(ages_prop, float)
                
                    for key in moved_keys:
                        self.accept_counts[key] += 1
                
                    _register(idx, "accepted")
                else:
                    _register(idx, "mh_reject")

            if i > self.burn_in and ((i - self.burn_in) % self.store_thin == 0):
                if sample_index < keep:
                    Ages_store[sample_index, :]            = ages_cur
                    Initial_Th_mean_store[sample_index, :] = theta[0]
                    U234_ratios_store[sample_index, :]     = theta[1]
                    Th232_ratios_store[sample_index, :]    = theta[3]
                    Th230_ratios_store[sample_index, :]    = theta[2]
                    posterior_store[sample_index]          = logp_cur
                    U234_initial_store[sample_index, :]    = 1.0 + ((theta[1] - 1.0) * np.exp(self.U234_lam * ages_cur))
                    sample_index += 1

            if (chain_id == 0) and (i % 5000 == 0):
                _print_diag(i)

            if i > 50 and i % 1000 == 0:
                self.Save_Parameters_and_Tuning(theta, chain_id)

            if i > 50 and (i % 2000 == 0) and (i < self.burn_in):
                _adapt_window(target=0.30, min_props_block=20, min_props_single=3)

        elapsed = time.time() - start_time
        if chain_id == 0:
            print(f"\nChain {chain_id} finished in {elapsed/60:.2f} min. Stored {sample_index} samples.\n")

        return (
            Ages_store,
            Initial_Th_mean_store,
            U234_ratios_store,
            Th232_ratios_store,
            Th230_ratios_store,
            posterior_store,
            U234_initial_store,
                )

    # ---------------- init/run helpers ----------------
    def check_starting_parameters(self):
        if self.Start_from_pickles:
            loaded = []
            for chain_id in range(self.n_chains):
                full_path = self.save_dir / f'{self.sample_name}_theta_{chain_id}.pkl'
                if full_path.exists():
                    with open(full_path, "rb") as f:
                        loaded.append(pickle.load(f))
                else:
                    break
            if len(loaded) == self.n_chains:
                print("Loaded starting θ from pickles")
                return loaded

        print("Generating new starting θ’s")
        thetas, found = self.Initial_Guesses_for_Model()
        if not np.all(found):
            bad = np.where(~found)[0].tolist()
            raise RuntimeError(
                f"No valid starting θ found for chains {bad}. "
                f"Try increasing max_attempts, relaxing priors/bounds, or reducing Age_Maximum."
            )
        return thetas

    def Run_MCMC(self):
        all_thetas = self.check_starting_parameters()
        for cid, th in enumerate(all_thetas):
            if th is None or (not isinstance(th, (tuple, list))) or len(th) != 4:
                raise ValueError(f"Bad starting θ for chain {cid}: {th}")

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs=-1)(
            delayed(run_chain)(all_thetas[chain_id], chain_id)
            for chain_id in range(self.n_chains)
        )

        self.Chain_Results = results
        return self.Chain_Results


    # ---------------- results utilities ----------------
    def Ensure_Chain_Results(self):
        if self.Chain_Results is None:
            self.Chain_Results = self.Run_MCMC()
        return self.Chain_Results

    def Get_Results_Dictionary(self):
        N_outputs = 7
        Results_ = self.Chain_Results
        if Results_ is None:
            self.Chain_Results = self.Run_MCMC()
            Results_ = self.Chain_Results

        z_vars = [f"z{i+1}" for i in range(N_outputs)]
        results_dict = []
        for chain_id, result in enumerate(Results_, start=1):
            result_dict = {}
            for var_name, value in zip(z_vars, result):
                result_dict[f"{var_name}_{chain_id}"] = value
            results_dict.append(result_dict)
        return results_dict

    def Get_Posterior_plot(self):
        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            log_p.append(chain_dict[f"z6_{i}"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(self.n_chains):
            ax.plot(log_p[i], label=f'Chain {i + 1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Posterior')
        ax.legend(frameon=True, loc=4, fontsize=10, ncol=2)

    def Get_Posterior_Values(self):
        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            log_p.append(chain_dict[f"z6_{i}"])
        return log_p

    def _summarize_draws(self, draws, center="median", lower=0.0):
        X = np.asarray(draws, float)
        if X.ndim == 1:
            X = X[np.isfinite(X)]
        elif X.ndim == 2:
            X = X[np.isfinite(X).all(axis=1)]
        else:
            raise ValueError("draws must be 1D or 2D")
        if X.size == 0:
            raise ValueError("No finite draws to summarize")

        if center.lower() == "median":
            c = np.percentile(X, 50.0, axis=0)
        elif center.lower() == "mean":
            c = np.mean(X, axis=0)
        else:
            raise ValueError("center must be 'median' or 'mean'")

        def et_bounds(conf):
            alpha = 100.0 * (1.0 - conf)
            lo = np.percentile(X, alpha / 2.0, axis=0)
            hi = np.percentile(X, 100.0 - alpha / 2.0, axis=0)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            if lower is not None:
                lo = np.maximum(lo, lower)
                hi = np.maximum(hi, lower)
                c2 = np.maximum(c, lower)
            else:
                c2 = c
            return c2, lo, hi

        _, lo68, hi68 = et_bounds(0.68)
        _, lo95, hi95 = et_bounds(0.95)
        p16 = np.percentile(X, 15.865525393145708, axis=0)
        p84 = np.percentile(X, 84.13447460685429, axis=0)
        sigma1 = np.maximum(0.5 * (p84 - p16), 0.0)

        c_out = np.maximum(c, lower) if lower is not None else c
        return c_out, (lo68, hi68), (lo95, hi95), sigma1

    def Get_Useries_Ages(self):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z1_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)
        return self._summarize_draws(all_draws, center="median", lower=0.0)

    def Get_Initial_Thoriums(self, return_sigma=False):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z2_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)
        return self._summarize_draws(all_draws, center="median", lower=0.0)

    def Get_234U_initial(self, return_sigma=False):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z7_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)
        return self._summarize_draws(all_draws, center="median", lower=0.0)


    # Chain diagnostics
    def Chain_Diagnostic_Vector(self, param = 'z1'):
        # Get chain stuffs
        results_dicts = self.Get_Results_Dictionary()
        samples = np.zeros([self.n_chains, results_dicts[0][f'{param}_1'].shape[0],
                           results_dicts[0][f'{param}_1'].shape[1]])
    
        for i in range(self.n_chains):
            samples[i,:, :] = results_dicts[i][f'{param}_{i+1}']
    
        m, n , d = samples.shape
        if m < 2:
            raise ValueError("Need a minimum of 2 chains")
        
        if n < 2:
            raise ValueError("Need a minimum of 2 samples")
        
        # Caculate an rhat value
        # Gets at how variable the between and within chain
        # state is.
        # This is more of a rule of thumb but is good as a guide for
        # the output
        chain_means = np.mean(samples, axis = 1)
        chain_vars = np.var(samples, axis = 1, ddof = 1)
        B = n * np.var(chain_means, axis = 0, ddof = 1)
        W = np.mean(chain_vars, axis = 0)
        var_hat = ((n - 1) / n) * W + (1/n) * B
        rhat = np.sqrt(var_hat / W)
            
        return rhat


    def Chain_diag_dataframe(self):

        inThor_rhat = self.Chain_Diagnostic_Vector(param = 'z2')
        r48_rhat = self.Chain_Diagnostic_Vector(param = 'z3')
        r28_rhat = self.Chain_Diagnostic_Vector(param = 'z4')
        r08_rhat = self.Chain_Diagnostic_Vector(param = 'z5')

        # Make a dataframe
        # Number samples from top to bottom (youngest to oldest)
        sample_idx = np.arange(inThor_rhat.size) + 1

        df = pd.DataFrame({"Sample index (top to bottom)": sample_idx,
                          "Initial thorium r-hat": inThor_rhat,
                          "234U_238U r-hat": r48_rhat,
                          "232Th_238U r-hat": r28_rhat,
                          "230Th_238U r-hat": r08_rhat,
                          })

        return df
        

    def MakeSummaryDataFrame(self):
        age_c, (age_lo68, age_hi68), (age_lo95, age_hi95), age_sigma = self.Get_Useries_Ages()
        U0_c,  (U0_lo68,  U0_hi68),  (U0_lo95,  U0_hi95),  U0_sigma  = self.Get_234U_initial()
        Th0_c, (Th0_lo68, Th0_hi68), (Th0_lo95, Th0_hi95), Th0_sigma = self.Get_Initial_Thoriums()

        df_all = pd.DataFrame({
            "Depth_Meas"     : self.data["Depths"].values,
            "Depth_Meas_err" : self.data["Depths_err"].values,

            "age"            : age_c,
            "age_lo68"       : age_lo68,
            "age_hi68"       : age_hi68,
            "age_lo95"       : age_lo95,
            "age_hi95"       : age_hi95,
            "age_err_sym"    : age_sigma,

            "initial thorium"      : Th0_c,
            "Th0_lo68"             : Th0_lo68,
            "Th0_hi68"             : Th0_hi68,
            "Th0_lo95"             : Th0_lo95,
            "Th0_hi95"             : Th0_hi95,
            "initial thorium err"  : Th0_sigma,
            "Th0_err_sym"          : Th0_sigma,

            "initial uranium"      : U0_c,
            "U0_lo68"              : U0_lo68,
            "U0_hi68"              : U0_hi68,
            "U0_lo95"              : U0_lo95,
            "U0_hi95"              : U0_hi95,
            "initial uranium err"  : U0_sigma,
            "U0_err_sym"           : U0_sigma,
        })

        df_all["age_err_lo68"] = np.maximum(age_c - age_lo68, 0.0)
        df_all["age_err_hi68"] = np.maximum(age_hi68 - age_c, 0.0)
        df_all["age_err_lo95"] = np.maximum(age_c - age_lo95, 0.0)
        df_all["age_err_hi95"] = np.maximum(age_hi95 - age_c, 0.0)

        df_all["Th0_err_lo68"] = np.maximum(Th0_c - Th0_lo68, 0.0)
        df_all["Th0_err_hi68"] = np.maximum(Th0_hi68 - Th0_c, 0.0)
        df_all["Th0_err_lo95"] = np.maximum(Th0_c - Th0_lo95, 0.0)
        df_all["Th0_err_hi95"] = np.maximum(Th0_hi95 - Th0_c, 0.0)

        df_all["U0_err_lo68"] = np.maximum(U0_c - U0_lo68, 0.0)
        df_all["U0_err_hi68"] = np.maximum(U0_hi68 - U0_c, 0.0)
        df_all["U0_err_lo95"] = np.maximum(U0_c - U0_lo95, 0.0)
        df_all["U0_err_hi95"] = np.maximum(U0_hi95 - U0_c, 0.0)

        return df_all
        
        
    def SummaryDataFrame(self):
        df_all = self.MakeSummaryDataFrame()
        output_path = self.save_dir /f"{self.sample_name}_ibis_summary.csv"
        df_all.to_csv(output_path, index=False)
        print(f"IBIS summary saved to: {output_path}")
        
