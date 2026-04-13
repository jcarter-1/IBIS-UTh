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

warnings.filterwarnings("ignore")

# ---------- fast scalar logpdf helpers (same math as scipy) ----------
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)

def _norm_logpdf_scalar(x, mu, sig):
    sig = max(float(sig), 1e-12)
    z = (float(x) - float(mu)) / sig
    return -0.5 * z * z - np.log(sig) - _LOG_SQRT_2PI

def _uniform_logpdf_scalar(x, a, b):
    x = float(x); a = float(a); b = float(b)
    if (x < a) or (x > b):
        return -np.inf
    return -np.log(b - a)


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
        
        # Save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()
        
        self.save_dir.mkdir(parents = True, exist_ok = True)        # Save directory
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
        self.TH0_MIN = 0.001
        self.TH0_MAX = self.compute_TH0_MAX(self.r08, self.r08_err, self.r28, self.r28_err)
        
        self.Th0_cdf_max = self._estimate_thor_xmax()
        self._thor_prior_grid = np.linspace(self.TH0_MIN, self.Th0_cdf_max, 4096)
        self._build_thor_inv_cdf()
        
        
        # tuning dict (per index)
        self.tuning = {}
        for i in range(self.N_meas):
            self.tuning[f'Initial_Thorium_{i}']     = 0.25
            self.tuning[f'Th230_U238_ratios_{i}']   = float(self.r08_err[i])
            self.tuning[f'Th232_U238_ratios_{i}']   = float(self.r28_err[i])
            self.tuning[f'U234_U238_ratios_{i}']    = float(self.r48_err[i])

        self.keys = list(self.tuning.keys())
        self.rng = np.random.default_rng()

        # ---- all-pairs indices for strat likelihood ----
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
            ("Initial_Thorium",       self.Initial_Thorium_Move),
            ("Th230_U238_ratios",     self.Th230_U238_Move),
            ("Th232_U238_ratios",     self.Th232_U238_Move),
            ("U234_U238_ratios",      self.U234_U238_Move),
            ("PerSampleBlock",        self.PerSampleBlock_Move),
            ("Smart_Order_Directed",  self.Smart_Order_Directed_Move),
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
        
    # =================
    # =================
    # All thorium stuff
    # =================
    # =================
    def thor_pdf(self, x):
        x = np.asarray(x, float)
        return np.asarray(self.Thor_KDE(x), float)
        
    def thor_logpdf(self, x):
        vals = np.asarray(self.thor_pdf(x), float)
        vals = np.clip(vals, 1e-300, None)   # avoid log(0)
        return np.log(vals)
        
    def _estimate_thor_xmax(self, x_min=0.0, q=0.9995, n_try=20000, max_rounds=8):
        """
        Estimate a sensible upper support bound for plotting / inverse-CDF sampling.
        """
        # Generic fallback: expand until tail density is negligible
        x_hi = 5
        for _ in range(max_rounds):
            grid = np.linspace(x_min, x_hi, 1024)
            pdf = self.thor_pdf(grid)
            peak = np.nanmax(pdf)
            tail = pdf[-1]
            if np.isfinite(peak) and peak > 0 and tail / peak < 1e-6:
                return float(x_hi)
            x_hi *= 2.0

        return float(x_hi)
        
    
    def _build_thor_inv_cdf(self):
        x = np.asarray(self._thor_prior_grid, float)
        pdf = self.thor_pdf(x)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf = np.clip(pdf, 0.0, None)
        area = np.trapz(pdf, x)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Thorium prior pdf has zero or invalid area.")

        pdf = pdf / area

        dx = np.diff(x)
        cdf = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
        ])
        cdf[-1] = 1.0
        # make strictly increasing
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
        
    def _thor_rvs(self, n):
        """
        Sample n values fro the thorium
        prior using inverse-CDF sampling
        """
        n = int(n)
        u = self.rng.random(n)
        out = np.asarray(self._thor_inv_cdf(u), float)

        # optional hard clipping to support
        if hasattr(self, "TH0_MIN") and hasattr(self, "TH0_MAX"):
            out = np.clip(out, self.TH0_MIN, self.TH0_MAX)

        return out

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
        if np.isfinite(g0) and abs(g0) < 1e-14:
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

        # ---------- 4) no bracket => no root (or numeric pathologies) ----------
        # Pick ONE of these behaviors:

        # (A) recommended for MCMC: raise and catch upstream => treat as -inf
        #raise RuntimeError(
        #    f"Age solver failed: no root in [{amin}, {amax}]. "
        #    f"params: Th0={Th_initial}, Th232={Th232}, U234={U234}, Th230={Th230}, g0={g0}"
        #)

        # (B) if you insist on returning NaN:
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
        lp  = np.sum(norm.logpdf(U234, loc=self.r48, scale=s48))
        lp += np.sum(norm.logpdf(Th230, loc=self.r08, scale=s08))
        lp += np.sum(norm.logpdf(Th232, loc=self.r28, scale=s28))
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
        if theta is None or len(theta) != 4:
            return -np.inf

        th, U234, Th230, Th232 = theta
        th = np.asarray(th, float)
        U234 = np.asarray(U234, float)
        Th230 = np.asarray(Th230, float)
        Th232 = np.asarray(Th232, float)

        if np.any(~np.isfinite(th)) or np.any(th < self.TH0_MIN) or np.any(th > self.TH0_MAX):
            return -np.inf

        if (np.any(~np.isfinite(U234)) or np.any(U234 <= 0) or
            np.any(~np.isfinite(Th230)) or np.any(Th230 <= 0) or
            np.any(~np.isfinite(Th232)) or np.any(Th232 <= 0)):
            return -np.inf

        lp = 0.0
        # Total prior is the combination of the
        # measured ratios and initial ratio
        lp += float(np.sum(self.thor_logpdf(th)))
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

    def log_posterior_given_ages(self, theta, ages):
        if not self._theta_is_finite_positive(theta):
            return -np.inf
        ages = np.asarray(ages, float)
        if np.any(~np.isfinite(ages)) or np.any(ages <= 0):
            return -np.inf

        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        lp += self.age_max_penalty(ages, w=1e5, power=2)

        if not self._hard_strat_check_adj(ages, dt_min=None):
            lp += self.strat_hinge_penalty_adj(ages, k=50)

        ll = float(self.strat_likelihood(ages))
        if not np.isfinite(ll):
            return -np.inf

        sat = self._age_is_saturated(ages, frac=1e-7)
        if np.any(sat):
            lp += -10.0 * float(np.sum(sat))

        return float(lp + ll)

    # ===================== CACHES (speed, same math) =====================
    def _init_prior_cache(self, theta):
        th, U234, Th230, Th232 = theta
        N = self.N_meas
        self._lp_u = np.empty(N, float)
        self._lp_k = np.empty(N, float)
        self._lp_r = np.empty(N, float)

        for i in range(N):
            self._lp_u[i] = _uniform_logpdf_scalar(th[i], self.TH0_MIN, self.TH0_MAX)
            self._lp_k[i] = float(self.thor_logpdf(th[i]))
            self._lp_r[i] = (
                _norm_logpdf_scalar(U234[i], self.r48[i], self.r48_err[i]) +
                _norm_logpdf_scalar(Th230[i], self.r08[i], self.r08_err[i]) +
                _norm_logpdf_scalar(Th232[i], self.r28[i], self.r28_err[i])
            )

        self._lp_prior_total = float(np.sum(self._lp_u) + np.sum(self._lp_k) + np.sum(self._lp_r))

    def _prior_delta_one_index(self, theta_new, idx):
        """
        Update cached prior totals for a proposal that changes ONLY idx.
        Returns delta prior (new-old). Updates caches in-place.
        """
        thn, U234n, Th230n, Th232n = theta_new

        old_u = self._lp_u[idx]
        old_k = self._lp_k[idx]
        old_r = self._lp_r[idx]

        new_u = _uniform_logpdf_scalar(thn[idx], self.TH0_MIN, self.TH0_MAX)
        if not np.isfinite(new_u):
            return -np.inf
        new_k = float(self.thor_logpdf(thn[idx]))
        new_r = (
            _norm_logpdf_scalar(U234n[idx], self.r48[idx], self.r48_err[idx]) +
            _norm_logpdf_scalar(Th230n[idx], self.r08[idx], self.r08_err[idx]) +
            _norm_logpdf_scalar(Th232n[idx], self.r28[idx], self.r28_err[idx])
        )

        self._lp_u[idx] = new_u
        self._lp_k[idx] = new_k
        self._lp_r[idx] = new_r

        d = float((new_u + new_k + new_r) - (old_u + old_k + old_r))
        self._lp_prior_total += d
        return d

    def _pair_contrib(self, pair_idx, ages):
        """
        Return per-pair log-likelihood contributions for the subset pair_idx.
        Same math as strat_likelihood(), just localized.
        """
        pair_idx = np.asarray(pair_idx, dtype=int)
        I = self._IJ_I[pair_idx]
        J = self._IJ_J[pair_idx]
        dA = ages[J] - ages[I]
        sig = np.maximum(self._pair_sigma[pair_idx], 1e-12)

        same = self._pair_same[pair_idx]
        out = np.empty(pair_idx.shape[0], float)

        if np.any(same):
            z = dA[same] / sig[same]
            out[same] = (-0.5 * z * z - np.log(sig[same]) - _LOG_SQRT_2PI)

        diff = ~same
        if np.any(diff):
            out[diff] = log_ndtr(dA[diff] / sig[diff])

        return out

    def _init_strat_cache(self, ages):
        self._ll_strat_total = float(self.strat_likelihood(ages))

    def _strat_delta_one_index(self, ages_old, ages_new, idx):
        pair_idx = self._pairs_by_idx[idx]
        if pair_idx.size == 0:
            return 0.0
        old = float(self._pair_contrib(pair_idx, ages_old).sum())
        new = float(self._pair_contrib(pair_idx, ages_new).sum())
        d = new - old
        self._ll_strat_total += float(d)
        return float(d)

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
                Th_initial = self._thor_rvs(self.N_meas)
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
                print(f"⚠️ chain {chain} failed after {max_attempts} attempts")
        if not np.all(found):
            bad = np.where(~found)[0].tolist()
            raise RuntimeError(
                f"No valid starting θ found for chains {bad}. "
                f"Likely: Age_Maximum too tight / Th0 bounds too restrictive / "
                f"data imply no root for some points."
            )
        return initial_thetas, found

    # ---------------- moves ----------------
    def Th232_U238_Move(self, theta, tuning, index):
        _, _, _, Th232 = theta
        Th232_prime = Th232.copy()
        y = np.log(max(Th232[index], 1e-12))
        Th232_prime[index] = float(np.exp(y + self.rng.normal(0.0, tuning)))
        return Th232_prime, True

    def Th230_U238_Move(self, theta, tuning, index):
        _, _, Th230, _ = theta
        Th230_prime = Th230.copy()
        y = np.log(max(Th230[index], 1e-12))
        Th230_prime[index] = float(np.exp(y + self.rng.normal(0.0, tuning)))
        return Th230_prime, True

    def U234_U238_Move(self, theta, tuning, index):
        _, U234, _, _ = theta
        U234_prime = U234.copy()
        y = np.log(max(U234[index], 1e-12))
        U234_prime[index] = float(np.exp(y + self.rng.normal(0.0, tuning)))
        return U234_prime, True

    def Initial_Thorium_Move(self, theta, tuning, index):
        init_th, _, _, _ = theta
        init_th_prime = init_th.copy()
        if self.rng.random() < 0.05:
            init_th_prime[index] = float(self._thor_rvs(1)[0])
        else:
            mu = np.log(max(init_th[index], 1e-12))
            init_th_prime[index] = float(np.exp(mu + self.rng.normal(0.0, tuning)))

        init_th_prime[index] = float(np.clip(init_th_prime[index], self.TH0_MIN, self.TH0_MAX))
        return init_th_prime, True

    def PerSampleBlock_Move(self, theta, _tuning_unused, index):
        th_cur, U234_cur, Th230_cur, Th232_cur = theta

        def _scale(key, default):
            return float(self.tuning.get(key, default))

        k_th   = f'Initial_Thorium_{index}'
        k_u234 = f'U234_U238_ratios_{index}'
        k_t30  = f'Th230_U238_ratios_{index}'
        k_t32  = f'Th232_U238_ratios_{index}'

        s_th   = _scale(k_th,   0.05 * max(1e-6, th_cur[index]))
        s_u234 = _scale(k_u234, 0.002)
        s_t30  = _scale(k_t30,  0.002)
        s_t32  = _scale(k_t32,  0.002)

        d_th   = self.rng.normal(0.0, s_th)
        d_u234 = self.rng.normal(0.0, s_u234)
        d_t30  = self.rng.normal(0.0, s_t30)
        d_t32  = self.rng.normal(0.0, s_t32)

        th_new    = th_cur.copy();    th_new[index]    = max(1e-12, th_cur[index]    + d_th)
        U234_new  = U234_cur.copy();  U234_new[index]  = max(1e-6,  U234_cur[index]  + d_u234)
        Th230_new = Th230_cur.copy(); Th230_new[index] = max(1e-12, Th230_cur[index] + d_t30)
        Th232_new = Th232_cur.copy(); Th232_new[index] = max(1e-12, Th232_cur[index] + d_t32)
        th_new[index] = float(np.clip(th_new[index], self.TH0_MIN, self.TH0_MAX))

        return (th_new, U234_new, Th230_new, Th232_new), None

    def _dage_dlogth(self, idx, th, U234, Th230, Th232, age0, eps=1e-3):
        amax = float(self.Age_Solve_Max)
        th_p = th * np.exp(+eps)
        th_m = th * np.exp(-eps)
        ap = self._solve_age_single(th_p, Th232, U234, Th230, a0=age0, amax=amax)
        am = self._solve_age_single(th_m, Th232, U234, Th230, a0=age0, amax=amax)
        if (not np.isfinite(ap)) or (not np.isfinite(am)):
            return np.nan
        return (ap - am) / (2 * eps)

    def Smart_Order_Directed_Move(self, theta, _tuning_ignored=None, _index_ignored=None, drift_scale=1.0):
        init_th, U234, Th230, Th232 = theta
        ages_here = self._ages_cache
        if ages_here is None or np.any(~np.isfinite(ages_here)):
            ages_here = self.ages_vector(init_th, U234, Th230, Th232)

        i, j, dA, z = self._worst_adj_violation(ages_here)

        if i is None:
            idx = int(self.rng.integers(0, self.N_meas))
            step = self.rng.normal(0.0, self.tuning.get(f"Initial_Thorium_{idx}", 0.2))
            th_new = init_th.copy()
            th_new[idx] = float(np.clip(np.exp(np.log(th_new[idx]) + step), self.TH0_MIN, self.TH0_MAX))
            return th_new, idx, np.log(init_th[idx]), np.log(th_new[idx]), float(self.tuning.get(f"Initial_Thorium_{idx}", 0.2)), 0, float(drift_scale)

        si = float(self.Age_Uncertainties[i]); sj = float(self.Age_Uncertainties[j])
        idx = j if (sj > si) else i
        want_age_up = (idx == j)

        target = max(abs(z) * float(np.max(self._adj_sigma)), 0.25 * max(si, sj))
        dA_target = +target if want_age_up else -target

        th0 = float(init_th[idx])
        dAdy = self._dage_dlogth(idx, th0, U234[idx], Th230[idx], Th232[idx], ages_here[idx])

        y_old = float(np.log(th0))
        sig = float(self.tuning.get(f'Initial_Thorium_{idx}', 0.2))

        if (not np.isfinite(dAdy)) or abs(dAdy) < 1e-6:
            s = -1.0 if want_age_up else +1.0
            y_new = float(self.rng.normal(y_old + s * drift_scale * sig, sig))
        else:
            step = float(np.clip(dA_target / dAdy, -1.0, +1.0))
            y_new = float(self.rng.normal(y_old + step, 0.5 * sig))

        th_new = init_th.copy()
        th_new[idx] = float(np.clip(np.exp(y_new), self.TH0_MIN, self.TH0_MAX))

        s_fwd = np.sign(y_new - y_old)
        return th_new, idx, y_old, y_new, sig, float(s_fwd), float(drift_scale)


    # ---------------- tuning adaptation ----------------
    def adapt_one(self, key, rate, t, target=0.25, smin=1e-4, smax=2.0):
        eta = 1.0 / np.sqrt(max(1, t))
        log_s = np.log(max(float(self.tuning[key]), 1e-12))
        log_s += eta * (rate - target)
        self.tuning[key] = float(np.clip(np.exp(log_s), smin, smax))

    # ---------------- I/O ----------------
    def Save_Parameters_and_Tuning(self, theta, chain_id):
    
        tf_file =  self.save_dir / f'tuning_{self.sample_name}_{chain_id}.pkl'
        theta_file = self.save_dir / f'{self.sample_name}_theta_{chain_id}.pkl'
        with open(theta_file, 'wb') as f:
            pickle.dump(theta, f)
        with open(tf_file, 'wb') as f:
            pickle.dump(self.tuning, f)

    # ===================== MAIN MCMC (optimized + fixed rollback) =====================
    def MCMC(self, theta, iterations, chain_id):
        start_time = time.time()
        Ndata = self.N_meas
        total_iterations = int(iterations) + int(self.burn_in)

        # load per-chain tuning
        tf_file = self.save_dir / f'tuning_{self.sample_name}_{chain_id}.pkl'
        if tf_file.exists() and self.Start_from_pickles:
            with open(tf_file, 'rb') as f:
                self.tuning = pickle.load(f)

        # counters
        self.proposal_counts = {k: 0 for k in self.keys}
        self.accept_counts   = {k: 0 for k in self.keys}
        self._rej = {"invalid_theta": 0, "age_nan": 0, "lp_nan": 0, "accepted": 0}
        self._last_prop_counts = {k: 0 for k in self.keys}
        self._last_acc_counts  = {k: 0 for k in self.keys}
        if not hasattr(self, "_adapt_step"):
            self._adapt_step = 0

        # initial eval (full) — IMPORTANT: monotone age guess
        init_ages_guess = np.linspace(0.05, 0.95, Ndata) * float(self.Age_Maximum)
        logp_cur, ages_cur = self.log_posterior(theta, ages_prev=init_ages_guess)
        self._ages_cache = ages_cur

        if (not np.isfinite(logp_cur)) or np.any(~np.isfinite(ages_cur)):
            raise RuntimeError(
                f"Initial theta produced non-finite posterior/ages for chain {chain_id}. "
                f"logp={logp_cur}, finite_ages={np.isfinite(ages_cur).all()}"
            )
        # init caches (for single-index fast updates)
        self._init_prior_cache(theta)
        self._init_strat_cache(ages_cur)
        # allocate storage
        keep = iterations // self.store_thin + int(iterations % self.store_thin != 0)
        Ages_store            = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Initial_Th_mean_store = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_initial_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_ratios_store     = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th232_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th230_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        posterior_store       = np.zeros(keep, dtype=self.store_dtype)

        sample_index = 0
        amax_single = float(self.Age_Solve_Max)

        # progress bar
        pbar = tqdm(
            range(1, total_iterations + 1),
            desc=f"Chain {chain_id}",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            disable=(chain_id != 0),
        )

        def _adapt_window(target=0.25, min_props=20):
            self._adapt_step += 1
            t = self._adapt_step
            for key in self.tuning:
                p_tot = self.proposal_counts.get(key, 0)
                a_tot = self.accept_counts.get(key, 0)
                p_last = self._last_prop_counts.get(key, 0)
                a_last = self._last_acc_counts.get(key, 0)
                dp = p_tot - p_last
                da = a_tot - a_last
                if dp >= min_props:
                    rate = da / dp
                    self.adapt_one(key, rate, t, target=target, smin=1e-6, smax=2.0)
                self._last_prop_counts[key] = p_tot
                self._last_acc_counts[key]  = a_tot

        def _print_diag(i):
            props = sum(self.proposal_counts.values()) or 1
            accs  = sum(self.accept_counts.values())
            acc_rate = accs / props
            ii, jj, dA, zz = self._worst_adj_violation(ages_cur)
            worst = "none" if ii is None else f"i={ii}, j={jj}, Δ={dA:.6g}, z={zz:.6g}"
            sat_frac = float(np.mean(self._age_is_saturated(ages_cur, frac=1e-7)))

            def _rate(prefix):
                p = sum(v for k, v in self.proposal_counts.items() if k.startswith(prefix))
                a = sum(v for k, v in self.accept_counts.items()   if k.startswith(prefix))
                return (a / p) if p else np.nan

            msg = (
                f"\n[i={i}] acc={acc_rate:.3f}  "
                f"rej(invθ)={self._rej['invalid_theta']} "
                f"rej(age_nan)={self._rej['age_nan']} "
                f"rej(lp_nan)={self._rej['lp_nan']} "
                f"accepted={self._rej['accepted']} "
                f"sat_frac={sat_frac:.3f}\n"
                f"worst_adj_violation: {worst}\n"
                f"acc-by-family: Th0={_rate('Initial_Thorium_'):.3f}  "
                f"U234={_rate('U234_U238_ratios_'):.3f}  "
                f"Th230={_rate('Th230_U238_ratios_'):.3f}  "
                f"Th232={_rate('Th232_U238_ratios_'):.3f}\n"
            )
            print(msg)

        # ---------------- main loop ----------------
        for i in pbar:
            rollback_pack = None  # IMPORTANT: reset every iteration

            move_name, move_func = random.choice(self._move_funcs)

            if move_name == "Smart_Order_Directed":
                (new_init_th, idx_used, y_old, y_new, sig, s_fwd, drift_scale) = move_func(theta, None, None)
                counter_key = f'Initial_Thorium_{idx_used}'
                self.proposal_counts[counter_key] = self.proposal_counts.get(counter_key, 0) + 1

                th_cur, U234_cur, Th230_cur, Th232_cur = theta
                theta_prop = (new_init_th, U234_cur, Th230_cur, Th232_cur)
                idx_moved = int(idx_used)

            else:
                idx = int(self.rng.integers(0, Ndata))
                counter_key = f'{move_name}_{idx}'
                self.proposal_counts[counter_key] = self.proposal_counts.get(counter_key, 0) + 1

                new_piece, _ = move_func(theta, self.tuning.get(counter_key, 0.0), idx)
                th_cur, U234_cur, Th230_cur, Th232_cur = theta

                if move_name == 'Initial_Thorium':
                    theta_prop = (new_piece, U234_cur, Th230_cur, Th232_cur)
                    idx_moved = idx
                elif move_name == 'U234_U238_ratios':
                    theta_prop = (th_cur, new_piece, Th230_cur, Th232_cur)
                    idx_moved = idx
                elif move_name == 'Th232_U238_ratios':
                    theta_prop = (th_cur, U234_cur, Th230_cur, new_piece)
                    idx_moved = idx
                elif move_name == 'Th230_U238_ratios':
                    theta_prop = (th_cur, U234_cur, new_piece, Th232_cur)
                    idx_moved = idx
                elif move_name == 'PerSampleBlock':
                    theta_prop = new_piece
                    idx_moved = None  # multi-var change: use full eval
                else:
                    theta_prop = theta
                    idx_moved = None

            # ---------- proposal evaluation ----------
            if idx_moved is not None:
                ages_prop = ages_cur.copy()
                th_p, U234_p, Th230_p, Th232_p = theta_prop

                if (th_p[idx_moved] <= 0) or (U234_p[idx_moved] <= 0) or (Th230_p[idx_moved] <= 0) or (Th232_p[idx_moved] <= 0):
                    logp_prop = -np.inf
                    self._rej["invalid_theta"] += 1

                else:
                    ages_prop[idx_moved] = self._solve_age_single(
                        th_p[idx_moved], Th232_p[idx_moved], U234_p[idx_moved], Th230_p[idx_moved],
                        a0=ages_cur[idx_moved], amax=amax_single
                    )

                    if not np.isfinite(ages_prop[idx_moved]):
                        logp_prop = -np.inf
                        self._rej["age_nan"] += 1
                    else:
                        # Save cache state for rollback if reject
                        rollback_pack = (
                            float(self._lp_prior_total),
                            float(self._ll_strat_total),
                            float(self._lp_u[idx_moved]),
                            float(self._lp_k[idx_moved]),
                            float(self._lp_r[idx_moved]),
                        )

                        # 1) prior delta (updates caches)
                        dprior = self._prior_delta_one_index(theta_prop, idx_moved)
                        if not np.isfinite(dprior):
                            logp_prop = -np.inf
                            # rollback immediately
                            lpP, llS, ou, ok, orr = rollback_pack
                            self._lp_prior_total = lpP
                            self._ll_strat_total = llS
                            self._lp_u[idx_moved] = ou
                            self._lp_k[idx_moved] = ok
                            self._lp_r[idx_moved] = orr
                            rollback_pack = None
                        else:
                            # 2) strat delta (updates cache)
                            self._strat_delta_one_index(ages_cur, ages_prop, idx_moved)

                            # 3) penalties exactly as before
                            lp_prop = float(self._lp_prior_total) + float(self._ll_strat_total)
                            lp_prop += self.age_max_penalty(ages_prop, w=1e5, power=2)
                            if not self._hard_strat_check_adj(ages_prop, dt_min=None):
                                lp_prop += self.strat_hinge_penalty_adj(ages_prop, k=50)
                            sat = self._age_is_saturated(ages_prop, frac=1e-7)
                            if np.any(sat):
                                lp_prop += -10.0 * float(np.sum(sat))

                            logp_prop = float(lp_prop)
                            if not np.isfinite(logp_prop):
                                self._rej["lp_nan"] += 1
                                # rollback immediately
                                lpP, llS, ou, ok, orr = rollback_pack
                                self._lp_prior_total = lpP
                                self._ll_strat_total = llS
                                self._lp_u[idx_moved] = ou
                                self._lp_k[idx_moved] = ok
                                self._lp_r[idx_moved] = orr
                                rollback_pack = None

            else:
                # full eval for PerSampleBlock / multi-changes
                logp_prop, ages_prop = self.log_posterior(theta_prop, ages_prev=ages_cur)
                if not np.isfinite(logp_prop):
                    self._rej["lp_nan"] += 1

            # ---------- asymmetric MH correction for Smart_Order_Directed ----------
            log_q_fwd = 0.0
            log_q_rev = 0.0
            if (move_name == "Smart_Order_Directed") and np.isfinite(logp_prop):
                inv = 1.0 / max(sig, 1e-12)
                log_q_fwd = -0.5 * ((y_new - (y_old + s_fwd * drift_scale * sig)) * inv) ** 2 \
                            - np.log(max(sig, 1e-12)) - 0.5 * np.log(2 * np.pi)

                i2, j2, _, _ = self._worst_adj_violation(ages_prop)
                s_rev = 0 if (i2 is None) else self._desired_th_sign(i2, j2, idx_used)

                log_q_rev = -0.5 * ((y_old - (y_new + s_rev * drift_scale * sig)) * inv) ** 2 \
                            - np.log(max(sig, 1e-12)) - 0.5 * np.log(2 * np.pi)

            # ---------- accept/reject ----------
            accept = np.isfinite(logp_prop) and (
                np.log(self.rng.random()) < (logp_prop - logp_cur + (log_q_rev - log_q_fwd))
            )

            if accept:
                theta = theta_prop
                logp_cur = float(logp_prop)
                ages_cur = np.asarray(ages_prop, float)
                self.accept_counts[counter_key] = self.accept_counts.get(counter_key, 0) + 1
                self._rej["accepted"] += 1

                # accepted: caches already reflect new state for single-index moves
                rollback_pack = None

                # if accepted a full-eval move, refresh caches exactly
                if idx_moved is None:
                    self._init_prior_cache(theta)
                    self._init_strat_cache(ages_cur)
            else:
                # rejected single-index proposal: rollback caches
                if rollback_pack is not None:
                    lpP, llS, ou, ok, orr = rollback_pack
                    self._lp_prior_total = lpP
                    self._ll_strat_total = llS
                    self._lp_u[idx_moved] = ou
                    self._lp_k[idx_moved] = ok
                    self._lp_r[idx_moved] = orr
                    rollback_pack = None

            self._ages_cache = ages_cur

            # ---------- store ----------
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

            # diagnostics
            if (chain_id == 0) and (i % 5000 == 0):
                _print_diag(i)

            # periodic save
            if i > 50 and i % 1000 == 0:
                self.Save_Parameters_and_Tuning(theta, chain_id)

            # adapt during burn-in
            if i > 50 and (i % 1000 == 0) and (i < self.burn_in):
                _adapt_window(target=0.25, min_props=20)

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

    def SummaryDataFrame(self):
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
        output_path = self.save_dir /f"{self.sample_name}_ibis_summary.csv"
        df_all.to_csv(output_path, index=False)
        print(f"IBIS summary saved to: {output_path}")
