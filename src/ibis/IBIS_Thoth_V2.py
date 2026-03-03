from __future__ import annotations

import os
import pickle
import numpy as np

from scipy.stats import norm, lognorm, truncnorm
from scipy.special import log_ndtr
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize, minimize_scalar

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from pathlib import Path

# =============================================================================
# Prior object
# =============================================================================
class ThoriumPrior1D:
    """
    1D prior with:
      - pdf(x)
      - rvs(size=n) using inverse-CDF on the saved grid/pdf
    """
    def __init__(self, grid, pdf, rng=None):
        self.grid = np.asarray(grid, float)
        self.pdf_grid = np.asarray(pdf, float)
        self.rng = rng if rng is not None else np.random.default_rng()

        Z = np.trapz(self.pdf_grid, self.grid)
        if not np.isfinite(Z) or Z <= 0:
            raise ValueError("Prior pdf does not integrate to a positive finite value.")
        self.pdf_grid = self.pdf_grid / Z

        dx = np.diff(self.grid)
        if np.any(dx <= 0):
            raise ValueError("grid must be strictly increasing.")

        cdf = np.cumsum(self.pdf_grid[:-1] * dx)
        cdf = np.concatenate([[0.0], cdf])
        cdf = cdf / cdf[-1]

        self._inv_cdf = interp1d(
            cdf, self.grid,
            bounds_error=False,
            fill_value=(self.grid[0], self.grid[-1]),
        )

        self._pdf = interp1d(
            self.grid, self.pdf_grid,
            bounds_error=False,
            fill_value=(0.0, 0.0),
        )

    def pdf(self, x):
        x = np.asarray(x, float)
        return np.asarray(self._pdf(x), float)

    def rvs(self, size=1):
        u = self.rng.random(int(size))
        return np.asarray(self._inv_cdf(u), float)


# =============================================================================
# U-series age equation (single analysis)
# =============================================================================
class U_Series_Age_Equation:
    """
    Your current equation form.
    No decay constant uncertainty propagation.

    Important:
    - For EB shaping, pass initial_err=0 to prevent sigma-cheating.
    """
    def __init__(
        self,
        r08, r08_err,
        r28, r28_err,
        r48, r48_err,
        r02_initial, r02_initial_err,
        rho_28_48=0.0, rho_08_48=0.0, rho_08_28=0.0,
    ):
        self.r08 = float(r08)
        self.r08_err = float(r08_err)

        self.r28 = float(r28)
        self.r28_err = float(r28_err)

        self.r48 = float(r48)
        self.r48_err = float(r48_err)

        self.r02_initial = float(r02_initial)
        self.r02_initial_err = float(r02_initial_err)

        # Cheng et al. (2013)
        self.lambda_230 = 9.17055e-06
        self.lambda_234 = 2.82203e-06

        self.rho_28_48 = float(rho_28_48)
        self.rho_08_48 = float(rho_08_48)
        self.rho_08_28 = float(rho_08_28)

    def Age_Equation(self, T: float) -> float:
        T = float(T)

        A = self.r08
        B = 1.0 - np.exp(-self.lambda_230 * T) * (1.0 - self.r02_initial * self.r28)
        D = self.r48 - 1.0

        lam_diff = self.lambda_230 - self.lambda_234
        if abs(lam_diff) < 1e-20:
            return np.nan

        E = self.lambda_230 / lam_diff
        F = 1.0 - np.exp(-lam_diff * T)

        C = D * E * F
        return A - B - C

    def Age_solver(self, age_guess=1e4, tmin=0.0, tmax=5e5, expand=25,
                   root_tol=1e-10, obj_tol=1e-14):
        f = self.Age_Equation
        lo = float(tmin)
        hi = float(tmax)

        flo = f(lo)
        fhi = f(hi)

        # try to expand to find a bracket
        if np.isfinite(flo) and np.isfinite(fhi) and (np.sign(flo) == np.sign(fhi)):
            for _ in range(int(expand)):
                hi *= 2.0
                fhi = f(hi)
                if np.isfinite(fhi) and (np.sign(flo) != np.sign(fhi)):
                    break

        # bracketed root
        if np.isfinite(flo) and np.isfinite(fhi) and (np.sign(flo) != np.sign(fhi)):
            t = float(brentq(f, lo, hi, maxiter=300))
            if np.isfinite(t) and abs(f(t)) <= root_tol:
                return t
            return np.nan

        # otherwise minimize f(t)^2 on [lo, hi]
        def obj(t):
            val = f(float(t))
            return np.inf if not np.isfinite(val) else float(val * val)

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        tbest = float(res.x)
        fbest2 = float(res.fun)

        # NO ROOT -> return NaN
        if (not np.isfinite(fbest2)) or (fbest2 > obj_tol):
            return np.nan

        return tbest

    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        Age = self.Age_solver()

        lam_diff = self.lambda_234 - self.lambda_230
        if abs(lam_diff) < 1e-20:
            return Age, np.nan

        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = -(self.r48 - 1.0) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3

        if (not np.isfinite(df_dT)) or (abs(df_dT) < 1e-30):
            return Age, np.nan

        dt_dr08 = -1.0 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = -((self.lambda_230 / lam_diff) * (1.0 - np.exp(lam_diff * Age))) / df_dT

        J = np.array([dt_dr08, dt_dr28, dt_dr02, dt_dr48], float)

        cov = np.zeros((4, 4), float)
        cov[0, 0] = self.r08_err ** 2
        cov[1, 1] = self.r28_err ** 2
        cov[2, 2] = self.r02_initial_err ** 2
        cov[3, 3] = self.r48_err ** 2

        cov[0, 1] = cov[1, 0] = self.rho_08_28 * self.r08_err * self.r28_err
        cov[0, 3] = cov[3, 0] = self.rho_08_48 * self.r08_err * self.r48_err
        cov[1, 3] = cov[3, 1] = self.rho_28_48 * self.r28_err * self.r48_err

        var = float(J @ cov @ J.T)
        if (not np.isfinite(var)) or (var < 0):
            return Age, np.nan

        return Age, float(np.sqrt(var))


# =============================================================================
# EB prior builder
# =============================================================================
class IBIS_Thoth_Robust:
    DEFAULT_META = dict(
        # depth handling
        depth_increases_down=True,
        depth_tol=0.0,  # 0 => no rounding

        # boutique sampler
        fraction_det=0.2,
        r02_clip_lo=0.001,
        r02_clip_hi=1000.0,

        # r02 max bound from data (IMPORTANT)
        r02_max_k_sigma=2.0,  # use (r08+kσ)/(r28-kσ)

        # r02 relative uncertainty proposal
        r02_err_mode="halfnorm",   # "fixed"|"uniform"|"halfnorm"
        r02_rel_err=0.25,
        r02_rel_min=0.005,
        r02_rel_max=0.5,

        # validity gates
        age_sigma_gate=2.0,
        age_eps=1e-12,

        # sigma cheating prevention (EB): default False
        use_r02_in_sigma=True,

        # batch sampler controls
        batch_size=600,
        max_batches=400,
        keep_frac=0.08,
        keep_factor=8,

        # layer windows
        neighbors=1,

        # pair weighting
        pair_weight_mode="exp",   # "none"|"exp"|"power"
        pair_tau_layers=2.0,
        pair_alpha=1.0,
        min_pair_weight=1e-6,

        # prior mixture
        global_mass=0.7,
        n_samples_global=4000,
        n_samples_layer=2500,

        # KDE fit controls
        kde_beta=0.6,
        bw_grid=None,
        cv=5,
        bw_subsample=3000,
        grid_n=1200,
        hi_pct=99.9,
        floor_pdf=1e-12,
    )

    def __init__(self, data, age_max, file_name="FILENAME", depth_col="Depths", diction_meta=None, save_dir = None):
        self.data = data
        self.age_max = float(age_max)
        self.file_name = str(file_name)
        self.depth_col = str(depth_col)

        self.meta = dict(self.DEFAULT_META)
        if diction_meta is not None:
            self.meta.update(dict(diction_meta))

        # ratios
        self.r08 = self.data["Th230_238U_ratios"].to_numpy(float)
        self.r28 = self.data["Th232_238U_ratios"].to_numpy(float)
        self.r48 = self.data["U234_U238_ratios"].to_numpy(float)

        self.r08_err = self.data["Th230_238U_ratios_err"].to_numpy(float)
        self.r28_err = self.data["Th232_238U_ratios_err"].to_numpy(float)
        self.r48_err = self.data["U234_U238_ratios_err"].to_numpy(float)

        depths = self.data[self.depth_col].to_numpy(float)
        dtol = float(self.meta.get("depth_tol", 0.0))
        if dtol and dtol > 0:
            depths = np.round(depths / dtol) * dtol
        self.depths = depths

        # sort + build layers
        self._sort_by_depth()
        self._build_layers()

        # storage
        self.Thorium_prior = None
        self._thor_prior_grid = None
        self._thor_prior_pdf = None
        self._thor_prior_weights = None
        self._thor_prior_meta = None
        
        # store save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()

        self.save_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------
    # sorting + layers
    # --------------------------
    def _sort_by_depth(self):
        order = np.argsort(self.depths)
        if not bool(self.meta.get("depth_increases_down", True)):
            order = order[::-1]

        self.depths = self.depths[order]
        self.r08 = self.r08[order]
        self.r28 = self.r28[order]
        self.r48 = self.r48[order]
        self.r08_err = self.r08_err[order]
        self.r28_err = self.r28_err[order]
        self.r48_err = self.r48_err[order]

    def _build_layers(self):
        self.layer_depths = np.unique(self.depths)
        self.layer_to_indices = {d: np.where(self.depths == d)[0] for d in self.layer_depths}
        depth_to_id = {d: i for i, d in enumerate(self.layer_depths)}
        self.layer_id = np.array([depth_to_id[d] for d in self.depths], dtype=int)

    def indices_for_layer_window(self, layer_depth, neighbors=1):
        layer_depths = self.layer_depths
        k = np.where(layer_depths == layer_depth)[0]
        if k.size == 0:
            raise ValueError("layer_depth not found")
        k = int(k[0])

        lo = max(0, k - int(neighbors))
        hi = min(len(layer_depths) - 1, k + int(neighbors))

        idx = []
        for kk in range(lo, hi + 1):
            d = layer_depths[kk]
            idx.append(self.layer_to_indices[d])
        return np.unique(np.concatenate(idx))

    # --------------------------
    # r02 max bounds from data
    # --------------------------
    def _compute_uncorrected_age_per_analysis(self, age_guess=1e4):
        """
        Compute "uncorrected" ages per analysis using r02=0 (no initial Th correction).
        Cached after first call.

        Returns
        -------
        t_uncorr : (N,) array
        """
        if hasattr(self, "_t_uncorr_cache") and self._t_uncorr_cache is not None:
            return self._t_uncorr_cache

        N = len(self.r08)
        t_uncorr = np.full(N, np.nan, float)

        for i in range(N):
            try:
                U = U_Series_Age_Equation(
                    self.r08[i], self.r08_err[i],
                    self.r28[i], self.r28_err[i],
                    self.r48[i], self.r48_err[i],
                    0.0, 0.0,  # r02=0 uncorrected
                )
                t_uncorr[i] = float(U.Age_solver(age_guess=age_guess))
            except Exception:
                t_uncorr[i] = np.nan

        # Optional: clip to [0, age_max] for stability
        t_uncorr = np.where(np.isfinite(t_uncorr), np.clip(t_uncorr, 0.0, float(self.age_max)), np.nan)

        self._t_uncorr_cache = t_uncorr
        return t_uncorr

    def r02_hi_from_measured_230232_and_uncorr_age(
        self,
        idx=None,
        k_sigma=3.0,
        eps=1e-12,
        ) -> float:
        """
        Upper bound for r02_0 derived from measured (r08/r28) rolled back using
        the UNCORRECTED age for the analyses in `idx`.

        If idx is None -> uses all analyses.

        Formula (per analysis i):
          r02_hi_i = (r08/r28)_i * exp(lambda230 * t_uncorr_i)

        Conservative uncertainty:
          r08_hi = r08 + kσ*r08_err
          r28_lo = r28 - kσ*r28_err

        Returns
        -------
        hi : float
        """
        l230 = 9.17055e-06  # Cheng et al. (2013)
        eps = float(eps)
        k = float(k_sigma)

        if idx is None:
            idx = np.arange(len(self.r08), dtype=int)
        idx = np.asarray(idx, int)

        t_uncorr = self._compute_uncorrected_age_per_analysis()

        r08 = np.asarray(self.r08, float)
        r28 = np.asarray(self.r28, float)
        r08e = np.asarray(self.r08_err, float)
        r28e = np.asarray(self.r28_err, float)

        r08_hi = np.maximum(r08 + k * r08e, eps)
        r28_lo = np.maximum(r28 - k * r28e, eps)

        ok = (
            np.isfinite(t_uncorr[idx]) &
            np.isfinite(r08_hi[idx]) &
            np.isfinite(r28_lo[idx]) &
            (r28_lo[idx] > 0)
        )

        if not np.any(ok):
            return float(self.meta.get("r02_clip_hi", 1000.0))

        r_meas = r08_hi[idx][ok] / r28_lo[idx][ok]
        r0_upper = r_meas * np.exp(l230 * t_uncorr[idx][ok])

        hi = float(np.nanmax(r0_upper))

        # Your safety controls
        safety   = float(self.meta.get("r02_hi_safety_factor", 1.25))
        hard_cap = float(self.meta.get("r02_hi_hard_cap", 1e6))
        floor_hi = float(self.meta.get("r02_hi_floor", 10.0))

        hi = max(hi * safety, floor_hi)
        hi = min(hi, hard_cap)
        return hi

    def r02_hi_for_layer(self, layer_depth, neighbors=0, k_sigma=3.0):
        """
        Convenience: compute the bound for a specific layer (optionally include neighbors).
        """
        idx = self.indices_for_layer_window(layer_depth, neighbors=int(neighbors))
        return self.r02_hi_from_measured_230232_and_uncorr_age(idx=idx, k_sigma=k_sigma)

    def r02_hi_for_window_indices(self, idx, k_sigma=3.0):
        """
        Convenience: compute the bound for any window index list/array.
        """
        return self.r02_hi_from_measured_230232_and_uncorr_age(idx=np.asarray(idx, int), k_sigma=k_sigma)

    # --------------------------
    # boutique r02 proposal
    # --------------------------
    def detrital_sample(self, n):
        a, b = (0.0 - 0.8) / 0.4, np.inf
        return truncnorm(a=a, b=b, loc=0.8, scale=0.4).rvs(size=int(n))

    def all_thorium(self, n):
        # your constructed lognormal
        z = norm.ppf(0.99)
        C = np.log(0.8) - np.log(50.0)
        disc = z**2 - 4*C
        sigma = (-z + np.sqrt(disc)) / 2
        mu = np.log(0.8) + sigma**2
        return lognorm(s=sigma, scale=np.exp(mu)).rvs(size=int(n))

    def boutique_thoriums(self, n):
        n = int(n)
        fract = float(self.meta.get("fraction_det", 0.5))
        det = (np.random.rand(n) < fract)

        x = np.empty(n, float)
        x[det] = self.detrital_sample(det.sum())
        x[~det] = self.all_thorium((~det).sum())

        np.clip(
            x,
            float(self.meta.get("r02_clip_lo", 0.001)),
            float(self.meta.get("r02_clip_hi", 1000.0)),
            out=x
        )
        return x

    def boutique_thoriums_bounded(self, n, r02_hi, r02_lo=None):
        if r02_lo is None:
            r02_lo = float(self.meta.get("r02_clip_lo", 0.001))
        x = self.boutique_thoriums(n)
        np.clip(x, float(r02_lo), float(r02_hi), out=x)
        return x

    # --------------------------
    # r02 uncertainty proposal (with a ranking penalty)
    # --------------------------
    def sample_r02_err(self, r02):
        mode = str(self.meta.get("r02_err_mode", "halfnorm"))
        rel_scale = float(self.meta.get("r02_rel_err", 0.05))
        rel_min = float(self.meta.get("r02_rel_min", 0.005))
        rel_max = float(self.meta.get("r02_rel_max", 0.2))

        r02 = np.asarray(r02, float)
        n = r02.size

        if mode == "fixed":
            rel = np.full(n, rel_scale, float)
            logp = np.zeros(n, float)

        elif mode == "uniform":
            rel = np.random.uniform(rel_min, rel_max, size=n)
            logp = -np.log(rel_max - rel_min) * np.ones(n, float)

        elif mode == "halfnorm":
            s = float(rel_scale)
            rel = np.empty(n, float)
            filled = 0
            while filled < n:
                m = max(1000, (n - filled) * 3)
                cand = np.abs(np.random.normal(0.0, s, size=m))
                cand = cand[(cand >= rel_min) & (cand <= rel_max)]
                take = min(cand.size, n - filled)
                if take > 0:
                    rel[filled:filled + take] = cand[:take]
                    filled += take
            # unnormalized log-kernel (fine for ranking)
            logp = -0.5 * (rel / s) ** 2 - np.log(s)

        else:
            raise ValueError("meta['r02_err_mode'] must be 'fixed', 'uniform', or 'halfnorm'")

        e02 = rel * r02
        return e02, logp

    # --------------------------
    # ages for candidate r02
    # --------------------------
    def compute_ages_for_indices(self, idx, r02, r02_err_for_ageunc=0.0):
        idx = np.asarray(idx, int)
        ages = np.empty(idx.size, float)
        errs = np.empty(idx.size, float)

        for k, ii in enumerate(idx):
            U = U_Series_Age_Equation(
                self.r08[ii], self.r08_err[ii],
                self.r28[ii], self.r28_err[ii],
                self.r48[ii], self.r48_err[ii],
                float(r02), float(r02_err_for_ageunc),
            )
            a, s = U.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
            ages[k] = a
            errs[k] = s

        return ages, errs

    def _valid_ages(self, ages, errs):
        ages = np.asarray(ages, float)
        errs = np.asarray(errs, float)
        if not (np.all(np.isfinite(ages)) and np.all(np.isfinite(errs))):
            return False
        gate = float(self.meta.get("age_sigma_gate", 2.0))
        if np.any(ages + gate * errs < 0.0):
            return False
        gate = float(self.meta.get("age_sigma_gate", 2.0))
        if np.any(ages - gate * errs > self.age_max):
            return False
        return True

    # --------------------------
    # strat likelihood (layer-aware, weighted)
    # --------------------------
    def strat_loglik(self, ages, errs, layer_ids):
        order = np.argsort(layer_ids)
        a = np.asarray(ages, float)[order]
        e = np.asarray(errs, float)[order]
        L = np.asarray(layer_ids, int)[order]

        n = a.size
        if n < 2:
            return 0.0

        delta = a[None, :] - a[:, None]  # a_j - a_i
        sigma = np.sqrt(e[:, None]**2 + e[None, :]**2)
        sigma = np.maximum(sigma, float(self.meta.get("age_eps", 1e-12)))
        z = delta / sigma

        iu = np.triu_indices(n, k=1)
        same = (L[:, None] == L[None, :])[iu]
        del_u = delta[iu]
        sig_u = sigma[iu]
        z_u = z[iu]

        mode = str(self.meta.get("pair_weight_mode", "exp"))
        if mode == "none":
            ll = 0.0
            if np.any(same):
                ll += float(np.sum(norm.logpdf(del_u[same], 0.0, sig_u[same])))
            if np.any(~same):
                ll += float(np.sum(log_ndtr(z_u[~same])))
            return ll

        # distance weights by layer separation
        dL = (L[None, :] - L[:, None]).astype(float)[iu]  # >=0 in upper tri
        if mode == "exp":
            tau = float(self.meta.get("pair_tau_layers", 2.0))
            w = np.exp(-dL / max(tau, 1e-12))
        elif mode == "power":
            alpha = float(self.meta.get("pair_alpha", 1.0))
            w = 1.0 / (1.0 + dL) ** max(alpha, 1e-12)
        else:
            w = np.ones_like(dL)

        wmin = float(self.meta.get("min_pair_weight", 1e-6))
        keep = (w >= wmin)
        if not np.any(keep):
            return -np.inf

        same_k = same[keep]
        w_k = w[keep]
        del_k = del_u[keep]
        sig_k = sig_u[keep]
        z_k = z_u[keep]

        ll = 0.0
        if np.any(same_k):
            ll += float(np.sum(w_k[same_k] * norm.logpdf(del_k[same_k], 0.0, sig_k[same_k])))
        if np.any(~same_k):
            ll += float(np.sum(w_k[~same_k] * log_ndtr(z_k[~same_k])))
        return ll

    # --------------------------
    # MAIN sampler (bounded r02!)
    # --------------------------
    def sample_r02_distribution(self, idx=None, desired=2000, verbose=True):
        """
        Returns: r02_draws, r02_err_draws, weights
        Weights ∝ exp(strat_ll + logp_rel)

        Uses bounded r02 proposals:
          r02 <= min_i (r08_i + kσ)/(r28_i - kσ) over the window.
        """
        if idx is None:
            idx = np.arange(self.depths.size, dtype=int)
        idx = np.asarray(idx, int)

        # window-specific r02 upper bound
        k_sigma_bound = float(self.meta.get("r02_max_k_sigma", 2.0))
        r02_hi = self.r02_hi_from_measured_230232_and_uncorr_age(idx=idx, k_sigma=k_sigma_bound)


        desired = int(desired)
        keep_factor = int(self.meta.get("keep_factor", 8))
        target_keep = int(desired * keep_factor)

        batch_size = int(self.meta.get("batch_size", 600))
        max_batches = int(self.meta.get("max_batches", 400))
        keep_frac = float(self.meta.get("keep_frac", 0.08))
        use_r02_in_sigma = bool(self.meta.get("use_r02_in_sigma", False))

        layer_ids = self.layer_id[idx]

        kept_r02, kept_e02, kept_ll = [], [], []

        for b in range(max_batches):
            r02 = self.boutique_thoriums_bounded(batch_size, r02_hi=r02_hi)
            e02, logp_rel = self.sample_r02_err(r02)

            ll_total = np.full(batch_size, -np.inf, float)

            for j in range(batch_size):
                r02_err_for_ageunc = float(e02[j]) if use_r02_in_sigma else 0.0
                ages, errs = self.compute_ages_for_indices(idx, r02[j], r02_err_for_ageunc=r02_err_for_ageunc)

                if not self._valid_ages(ages, errs):
                    continue

                ll_strat = self.strat_loglik(ages, errs, layer_ids)
                ll_total[j] = float(ll_strat) + float(logp_rel[j])

            good = np.isfinite(ll_total)
            if not np.any(good):
                if verbose and (b % 10 == 0):
                    print(f"[thoth] batch {b+1}/{max_batches}: 0 valid (r02_hi={r02_hi:.4g})")
                continue

            good_idx = np.where(good)[0]
            kkeep = max(10, int(np.ceil(keep_frac * good_idx.size)))
            top_local = np.argsort(ll_total[good_idx])[-kkeep:]
            chosen = good_idx[top_local]

            kept_r02.extend(r02[chosen].tolist())
            kept_e02.extend(e02[chosen].tolist())
            kept_ll.extend(ll_total[chosen].tolist())

            if verbose and (b % 5 == 0):
                print(f"[thoth] batch {b+1}/{max_batches}: valid={good_idx.size} kept_total={len(kept_r02)} (r02_hi={r02_hi:.4g})")

            if len(kept_r02) >= target_keep:
                break

        kept_r02 = np.asarray(kept_r02, float)
        kept_e02 = np.asarray(kept_e02, float)
        kept_ll = np.asarray(kept_ll, float)

        if kept_r02.size == 0:
            return np.array([]), np.array([]), np.array([])

        w = np.exp(kept_ll - np.max(kept_ll))
        w /= w.sum()

        if kept_r02.size > desired:
            top = np.argsort(w)[-desired:]
            kept_r02 = kept_r02[top]
            kept_e02 = kept_e02[top]
            kept_ll = kept_ll[top]
            w = w[top]
            w /= w.sum()

        return kept_r02, kept_e02, w

    def sample_global(self, desired=None, verbose=True):
        if desired is None:
            desired = int(self.meta.get("n_samples_global", 4000))
        return self.sample_r02_distribution(idx=None, desired=int(desired), verbose=verbose)

    def sample_by_layer_windows(self, neighbors=None, desired=None, verbose=False):
        if neighbors is None:
            neighbors = int(self.meta.get("neighbors", 1))
        if desired is None:
            desired = int(self.meta.get("n_samples_layer", 2500))

        out = {}
        for d in self.layer_depths:
            idx = self.indices_for_layer_window(d, neighbors=int(neighbors))
            out[float(d)] = self.sample_r02_distribution(idx=idx, desired=int(desired), verbose=verbose)
        return out

    # --------------------------
    # KDE helpers
    # --------------------------
    @staticmethod
    def _effective_sample_size(w):
        w = np.asarray(w, float)
        w = w[np.isfinite(w)]
        if w.size == 0:
            return 0.0
        s = w.sum()
        if s <= 0:
            return 0.0
        w = w / s
        denom = np.sum(w**2)
        return float(1.0 / denom) if denom > 0 else 0.0

    def _select_bw_unweighted(self, x_log, bw_grid=None, cv=5):
        x_log = np.asarray(x_log, float).reshape(-1, 1)
        if x_log.shape[0] < 20:
            return 0.25

        if bw_grid is None:
            bw_grid = np.logspace(-2.2, 0.8, 30)

        n = x_log.shape[0]
        n_sub = int(self.meta.get("bw_subsample", 3000))
        if n > n_sub:
            take = np.random.choice(n, size=n_sub, replace=False)
            x_cv = x_log[take]
        else:
            x_cv = x_log

        cv_eff = min(int(cv), max(2, x_cv.shape[0] // 80))
        cv_eff = min(cv_eff, x_cv.shape[0])

        gs = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": bw_grid},
            cv=cv_eff,
            n_jobs=-1,
        )
        gs.fit(x_cv)
        return float(gs.best_params_["bandwidth"])

    def _fit_kde_fixed_bw(self, draws, bandwidth, weights=None, beta=0.6):
        draws = np.asarray(draws, float)
        ok = np.isfinite(draws) & (draws > 0)
        draws = draws[ok]

        x = np.log(draws)[:, None]
        kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))

        if weights is None:
            kde.fit(x)
            return kde

        w = np.asarray(weights, float)
        w = w[ok]
        w = np.clip(w, 0.0, np.inf)
        if w.sum() <= 0:
            kde.fit(x)
            return kde

        # tempered weights
        w = (w ** float(beta))
        w = w / w.sum()
        kde.fit(x, sample_weight=w)
        return kde

    # --------------------------
    # Build mixture prior
    # --------------------------
    def build_prior(self, verbose=True):
        neighbors = int(self.meta.get("neighbors", 1))
        global_mass = float(np.clip(self.meta.get("global_mass", 0.7), 0.0, 1.0))

        n_global = int(self.meta.get("n_samples_global", 4000))
        n_layer = int(self.meta.get("n_samples_layer", 2500))

        kde_beta = float(self.meta.get("kde_beta", 0.6))
        cv = int(self.meta.get("cv", 5))
        bw_grid = self.meta.get("bw_grid", None)

        grid_n = int(self.meta.get("grid_n", 1200))
        hi_pct = float(self.meta.get("hi_pct", 99.9))
        floor_pdf = float(self.meta.get("floor_pdf", 1e-12))

        # 1) global draws
        g_r, _g_e, g_w = self.sample_global(desired=n_global, verbose=verbose)
        global_ok = (np.asarray(g_r).size >= 20)
        if not global_ok:
            if verbose:
                print("[thoth] No global draws found -> using neighbors-only mixture.")
            global_mass = 0.0
            g_r = np.array([], float)
            g_w = np.array([], float)

        # 2) layer windows draws
        layer_dict = self.sample_by_layer_windows(neighbors=neighbors, desired=n_layer, verbose=False)

        # 3) pick ONE bandwidth on pooled draws
        bw_pool = []
        if global_ok:
            bw_pool.append(g_r[g_r > 0])
        for _, (r, _, _) in layer_dict.items():
            r = np.asarray(r, float)
            r = r[np.isfinite(r) & (r > 0)]
            if r.size:
                bw_pool.append(r)

        if len(bw_pool) == 0:
            raise ValueError("No draws available (global and layer windows both empty).")

        bw_pool = np.concatenate(bw_pool)
        bw = self._select_bw_unweighted(np.log(bw_pool), bw_grid=bw_grid, cv=cv)

        # 4) fit global KDE
        kdes = []
        mix_w = []

        if global_ok and global_mass > 0:
            kde_g = self._fit_kde_fixed_bw(g_r, bw, weights=g_w, beta=kde_beta)
            kdes.append(kde_g)
            mix_w.append(global_mass)

        # 5) fit layer KDEs with evidence weights ~ sqrt(ESS)
        layer_kdes = []
        layer_ev = []

        for d, (r, _e, w) in layer_dict.items():
            r = np.asarray(r, float)
            ok = np.isfinite(r) & (r > 0)
            r = r[ok]
            if r.size < 20:
                continue

            if w is None:
                w_use = None
                ess = float(r.size)
            else:
                w = np.asarray(w, float)
                w_use = w[ok]
                ess = self._effective_sample_size(w_use)

            if ess <= 5:
                continue

            kde_l = self._fit_kde_fixed_bw(r, bw, weights=w_use, beta=kde_beta)
            layer_kdes.append(kde_l)
            layer_ev.append(np.sqrt(ess))

        layer_mass = 1.0 - global_mass
        if layer_mass > 0 and len(layer_kdes) > 0:
            layer_ev = np.asarray(layer_ev, float)
            layer_ev = layer_ev / layer_ev.sum()
            layer_wts = layer_mass * layer_ev

            kdes.extend(layer_kdes)
            mix_w.extend(layer_wts.tolist())
        else:
            if global_ok and len(kdes) == 1:
                mix_w = [1.0]
            elif len(layer_kdes) > 0:
                layer_ev = np.asarray(layer_ev, float)
                layer_ev = layer_ev / layer_ev.sum()
                kdes = layer_kdes
                mix_w = layer_ev.tolist()
            else:
                raise ValueError("Failed to build any KDEs.")

        mix_w = np.asarray(mix_w, float)
        mix_w = mix_w / mix_w.sum()

        # 6) evaluation grid
        all_draws = []
        if global_ok:
            all_draws.append(g_r[g_r > 0])
        for _, (r, _, _) in layer_dict.items():
            r = np.asarray(r, float)
            r = r[np.isfinite(r) & (r > 0)]
            if r.size:
                all_draws.append(r)

        all_draws = np.concatenate(all_draws)
        lo = max(float(np.min(all_draws)), 1e-8)
        hi = float(np.percentile(all_draws, hi_pct))
        hi = max(hi*1.1, lo * 1.01)

        grid_log = np.linspace(np.log(lo), np.log(hi), int(grid_n))
        grid = np.exp(grid_log)

        # 7) evaluate mixture in log-space then back-transform
        X = grid_log[:, None]
        pdf_log = np.zeros_like(grid, float)
        for wi, kde in zip(mix_w, kdes):
            pdf_log += float(wi) * np.exp(kde.score_samples(X))

        pdf = pdf_log / grid  # Jacobian 1/r

        Z = float(np.trapz(pdf, grid))
        if (not np.isfinite(Z)) or (Z <= 0):
            pdf[:] = floor_pdf
        else:
            pdf /= Z
            pdf = np.maximum(pdf, floor_pdf)

        self._thor_prior_grid = grid
        self._thor_prior_pdf = pdf
        self._thor_prior_weights = mix_w
        self._thor_prior_meta = dict(
            neighbors=neighbors,
            global_mass=global_mass,
            bandwidth=bw,
            kde_beta=kde_beta,
            global_ok=bool(global_ok),
            layer_kde_count=int(len(layer_kdes)),
            meta=dict(self.meta),
        )

        # convenient callable
        self.Thorium_prior = interp1d(grid, pdf, bounds_error=False, fill_value=floor_pdf)
        return self.Thorium_prior

    # --------------------------
    # Save/load robustly (grid/pdf/meta), rebuild callable
    # --------------------------
    def save_prior_payload(self, fname=None):
        if self._thor_prior_grid is None or self._thor_prior_pdf is None:
            self.build_prior(verbose=False)

        if fname is None:
            fname = f"{self.file_name}_thor_prior_payload.pkl"

        payload = dict(
            grid=self._thor_prior_grid,
            pdf=self._thor_prior_pdf,
            weights=self._thor_prior_weights,
            meta=self._thor_prior_meta,
        )
        with open(fname, "wb") as f:
            pickle.dump(payload, f)

        print(f"Thorium prior payload saved to {fname}")
        return fname

    def load_prior_payload(self, fname=None):
        if fname is None:
            fname = f"{self.file_name}_thor_prior_payload.pkl"
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No prior payload found at: {fname}")

        with open(fname, "rb") as f:
            payload = pickle.load(f)

        self._thor_prior_grid = np.asarray(payload["grid"], float)
        self._thor_prior_pdf = np.asarray(payload["pdf"], float)
        self._thor_prior_weights = payload.get("weights", None)
        self._thor_prior_meta = payload.get("meta", None)

        floor_pdf = float(self.meta.get("floor_pdf", 1e-12))
        self.Thorium_prior = interp1d(self._thor_prior_grid, self._thor_prior_pdf, bounds_error=False, fill_value=floor_pdf)

        print(f"Thorium prior payload loaded from {fname}")
        return self.Thorium_prior

    def as_prior_object(self, rng=None):
        if self._thor_prior_grid is None or self._thor_prior_pdf is None:
            self.build_prior(verbose=False)
        return ThoriumPrior1D(self._thor_prior_grid, self._thor_prior_pdf, rng=rng)


    # --------------------------
    # save / load (stable payload)
    # --------------------------
    def save_thor_prior(self):
        if self.Thorium_prior is None or self._thor_prior_grid is None or self._thor_prior_pdf is None:
            self.build_prior(self.meta.get("Verbose", False))

        full_path = self.save_dir / f"{self.file_name}.pkl"

        # write the pickle
        with open(full_path, 'wb') as f:
            pickle.dump(self.Thorium_prior, f)
        print(f"Initial thorium composition prior saved to {full_path}")

        return full_path

    def load_thor_prior(self):
        # same filename we used for saving
        full_path = self.save_dir / f"{self.file_name}.pkl"

        if full_path.exists():
            with open(full_path, 'rb') as f:
                self.Thorium_prior = pickle.load(f)
            print(f"Initial thorium prior loaded from {full_name}")
        else:
            print(f"No prior at {full_name}, will compute and save now")
            self.save_thor_prior()
