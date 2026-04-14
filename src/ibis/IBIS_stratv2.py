import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed
import dill as pickle
from pathlib import Path
import os
import sys


class IBIS_Strat2:
    """
    Simple latent-age age-depth model.

    Model:
    - fixed depth grid
    - latent ages at grid nodes
    - monotone increasing with depth
    - weak smoothness prior
    - hybrid observation model:
        * half-normal from floor for effectively one-sided zero-age constraints
        * asymmetric Gaussian otherwise
    """

    def __init__(
        self,
        U_series_ages,
        U_series_ages_err_low,
        U_series_ages_err_high,
        data,
        sample_name="SAMPLE_NAME",
        Start_from_pickles=True,
        n_chains=3,
        iterations=30000,
        burn_in=10000,
        thin=10,
        resolution=100,
        top_age=None,
        top_age_err=None,
        top_age_depth=None,
        model_top_depth=None,
        model_bottom_depth=None,
        pad_frac=0.10,
        sigma_extra=0.0,
        smoothness=1e-7,
        save_dir=None,
    ):
        self.sample_name = str(sample_name)
        self.Start_from_pickles = bool(Start_from_pickles)
        self.n_chains = int(n_chains)
        self.iterations = int(iterations)
        self.burn_in = int(burn_in)
        self.thin = int(thin) if thin and thin > 0 else 10
        self.resolution = int(resolution)
        self.pad_frac = float(pad_frac)
        self.sigma_extra = float(max(sigma_extra, 0.0))
        self.smoothness = float(max(smoothness, 1e-12))
        self.Chain_Results = None

        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------
        # Observed ages
        # ---------------------------
        self.u_ages = np.asarray(U_series_ages, float).copy()
        self.low_age_err_raw = np.asarray(U_series_ages_err_low, float).copy()
        self.high_age_err_raw = np.asarray(U_series_ages_err_high, float).copy()

        # ---------------------------
        # Observed depths
        # ---------------------------
        if "Depth_Meas" in data.columns:
            self.Depths = np.asarray(data["Depth_Meas"].values, float).copy()
        elif "Depths" in data.columns:
            self.Depths = np.asarray(data["Depths"].values, float).copy()
        else:
            raise KeyError(
                f"No depth column found. Expected 'Depth_Meas' or 'Depths'. "
                f"Available columns: {list(data.columns)}"
            )

        if "Depth_Meas_err" in data.columns:
            self.Depths_err = np.asarray(data["Depth_Meas_err"].values, float).copy()
        elif "Depths_err" in data.columns:
            self.Depths_err = np.asarray(data["Depths_err"].values, float).copy()
        else:
            self.Depths_err = np.zeros_like(self.Depths, dtype=float)

        # ---------------------------
        # Optional top age anchor
        # ---------------------------
        if top_age is not None:
            if top_age_err is None:
                raise ValueError("If top_age is provided, top_age_err must also be provided.")
            if top_age_depth is None:
                raise ValueError("If top_age is provided, top_age_depth must also be provided.")

            self.u_ages = np.insert(self.u_ages, 0, float(top_age))
            self.low_age_err_raw = np.insert(self.low_age_err_raw, 0, float(top_age_err) / 2.0)
            self.high_age_err_raw = np.insert(self.high_age_err_raw, 0, float(top_age_err) / 2.0)
            self.Depths = np.insert(self.Depths, 0, float(top_age_depth))
            self.Depths_err = np.insert(self.Depths_err, 0, 1e-6)

        # ---------------------------
        # Safety floors
        # ---------------------------
        EPS = 1e-12
        self.Depths_err = np.maximum(np.asarray(self.Depths_err, float), EPS)

        # Sort by depth
        order = np.argsort(self.Depths)
        self.Depths = self.Depths[order]
        self.Depths_err = self.Depths_err[order]
        self.u_ages = self.u_ages[order]
        self.low_age_err_raw = self.low_age_err_raw[order]
        self.high_age_err_raw = self.high_age_err_raw[order]

        # Floored versions for numerical calculations
        self.low_age_err = np.maximum(self.low_age_err_raw, EPS)
        self.high_age_err = np.maximum(self.high_age_err_raw, EPS)

        # ---------------------------
        # Fixed latent depth grid
        # ---------------------------
        obs_top = float(np.min(self.Depths))
        obs_bottom = float(np.max(self.Depths))
        obs_span = max(obs_bottom - obs_top, 1e-9)

        default_top = max(0.0, obs_top - self.pad_frac * obs_span)
        default_bottom = obs_bottom + self.pad_frac * obs_span

        self.model_top_depth = float(default_top if model_top_depth is None else model_top_depth)
        self.model_bottom_depth = float(default_bottom if model_bottom_depth is None else model_bottom_depth)

        if self.model_top_depth >= self.model_bottom_depth:
            raise ValueError("model_bottom_depth must be greater than model_top_depth.")

        self.depth_grid = np.linspace(self.model_top_depth, self.model_bottom_depth, self.resolution)

        # ---------------------------
        # Age bounds
        # ---------------------------
        self.age_floor = 0.0
        i_bot = int(np.argmax(self.Depths))
        sig_bot = 0.5 * (self.low_age_err[i_bot] + self.high_age_err[i_bot])
        self.age_ceiling = float(self.u_ages[i_bot] + 5.0 * sig_bot)

        # Proposal weights
        self.p_local = 0.35
        self.p_block = 0.45
        self.p_tilt = 0.20

    # =========================================================
    # Helpers
    # =========================================================
    def _project_monotone(self, age, min_increment=1e-8):
        age = np.asarray(age, float).copy()
        age = np.clip(age, self.age_floor + min_increment, self.age_ceiling - min_increment)
        for i in range(1, age.size):
            age[i] = max(age[i], age[i - 1] + min_increment)
        return age

    def _initial_model(self):
        """
        Start from observed age-depth shape rather than a straight-line fit.
        """
        age0 = np.interp(self.depth_grid, self.Depths, self.u_ages)
        return self._project_monotone(age0)

    def _interp_model_to_obs(self, Age_Model):
        return np.interp(self.Depths, self.depth_grid, Age_Model)

    def _safe_percentiles(self, X, qs=(2.5, 50, 97.5)):
        X = np.asarray(X, float)
        if X.size == 0 or X.shape[0] == 0:
            raise ValueError("No posterior samples available.")
        return [np.percentile(X, q, axis=0) for q in qs]

    # =========================================================
    # Prior
    # =========================================================
    def _rw2_logprior(self, age):
        d2 = np.diff(age, n=2)
        if d2.size == 0:
            return 0.0
        return float(-0.5 * self.smoothness * np.dot(d2, d2))

    def Log_Priors(self, Age_Model):
        Age_Model = np.asarray(Age_Model, float)

        if np.any(~np.isfinite(Age_Model)):
            return -np.inf
        if np.any(np.diff(Age_Model) <= 0.0):
            return -np.inf
        if np.min(Age_Model) < self.age_floor:
            return -np.inf
        if np.max(Age_Model) > self.age_ceiling:
            return -np.inf

        # Only the structural prior matters here
        return self._rw2_logprior(Age_Model)

    # =========================================================
    # Likelihood
    # =========================================================
    def _halfnorm_logpdf_from_floor(self, x, floor, sigma):
        x = float(x)
        floor = float(floor)
        sigma = max(float(sigma), 1e-12)

        if x < floor:
            return -np.inf

        z = (x - floor) / sigma
        return 0.5 * np.log(2.0 / np.pi) - np.log(sigma) - 0.5 * z * z

    def _obs_logpdf(
        self,
        mu,
        sigma_low_raw,
        sigma_low_eff,
        sigma_high_eff,
        model,
        floor_tol=1e-10,
        asym_ratio=0.05,
    ):
        mu = float(mu)
        sigma_low_raw = float(max(sigma_low_raw, 0.0))
        sigma_low_eff = float(max(sigma_low_eff, 1e-12))
        sigma_high_eff = float(max(sigma_high_eff, 1e-12))
        model = float(model)

        # One-sided lower-bound age: use half-normal from the physical floor
        if (mu <= self.age_floor + floor_tol) and (sigma_low_raw < asym_ratio * sigma_high_eff):
            return self._halfnorm_logpdf_from_floor(model, self.age_floor, sigma_high_eff)

        # Otherwise: ordinary asymmetric Gaussian
        delta = mu - model
        sigma = sigma_low_eff if delta < 0.0 else sigma_high_eff
        return -0.5 * (delta / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)

    def Log_Likelihood(self, Age_Model):
        age_at_obs = self._interp_model_to_obs(Age_Model)

        slope = np.gradient(Age_Model, self.depth_grid)
        slope_at_obs = np.interp(self.Depths, self.depth_grid, slope)
        sigma_depth_age = np.abs(slope_at_obs) * self.Depths_err

        sigma_low_eff = np.sqrt(
            self.low_age_err**2 + sigma_depth_age**2 + self.sigma_extra**2 + 1e-12
        )
        sigma_high_eff = np.sqrt(
            self.high_age_err**2 + sigma_depth_age**2 + self.sigma_extra**2 + 1e-12
        )

        ll = 0.0
        for mu, sl_raw, sl_eff, sh_eff, mod in zip(
            self.u_ages,
            self.low_age_err_raw,
            sigma_low_eff,
            sigma_high_eff,
            age_at_obs,
        ):
            ll += self._obs_logpdf(mu, sl_raw, sl_eff, sh_eff, mod)

        return float(ll)

    def Log_Posterior(self, Age_Model):
        lp = self.Log_Priors(Age_Model)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.Log_Likelihood(Age_Model)
        if not np.isfinite(ll):
            return -np.inf
        return float(lp + ll)

    # =========================================================
    # Initialization
    # =========================================================
    def Initial_Guesses_for_Model(self):
        base = self._initial_model()
        sigma0 = max(1e-3, np.nanmean(0.5 * (self.low_age_err + self.high_age_err)))

        starts = []
        for _ in range(self.n_chains):
            trial = base + np.random.normal(0.0, 0.25 * sigma0, size=base.size)
            trial = self._project_monotone(trial)
            starts.append(trial)
        return starts

    # =========================================================
    # Proposals
    # =========================================================
    def _propose_local(self, Age_Model, scale):
        """
        Monotone-preserving local move: propose only within valid neighbors.
        """
        A = Age_Model.copy()
        k = np.random.randint(0, A.size)
        eps = 1e-8

        lower = self.age_floor + eps if k == 0 else A[k - 1] + eps
        upper = self.age_ceiling - eps if k == A.size - 1 else A[k + 1] - eps

        if upper <= lower:
            return A

        proposal = A[k] + np.random.normal(0.0, scale)
        A[k] = np.clip(proposal, lower, upper)
        return A

    def _propose_block(self, Age_Model, scale):
        """
        Segment move followed by monotone projection.
        Kept simple for now.
        """
        A = Age_Model.copy()
        n = A.size
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 1, min(n, i + max(3, n // 5)) + 1)
        A[i:j] += np.random.normal(0.0, scale)
        return self._project_monotone(A)

    def _propose_tilt(self, Age_Model, scale):
        """
        Ramp move across a segment followed by monotone projection.
        """
        A = Age_Model.copy()
        n = A.size
        i = np.random.randint(0, max(1, n - 2))
        j = np.random.randint(i + 2, min(n, i + max(4, n // 4)) + 1)
        amp = np.random.normal(0.0, scale)
        A[i:j] += amp * np.linspace(-1.0, 1.0, j - i)
        return self._project_monotone(A)

    def propose_state(self, Age_Model, scale):
        r = np.random.rand()
        if r < self.p_local:
            return self._propose_local(Age_Model, scale)
        elif r < self.p_local + self.p_block:
            return self._propose_block(Age_Model, scale)
        else:
            return self._propose_tilt(Age_Model, scale)

    # =========================================================
    # Paths
    # =========================================================
    def _theta_path(self, chain_id):
        return self.save_dir / f"{self.sample_name}_Age_theta_{chain_id}.pkl"

    def _tuning_path(self, chain_id):
        return self.save_dir / f"{self.sample_name}_Age_tuning_{chain_id}.pkl"

    def _meta_path(self):
        return self.save_dir / f"{self.sample_name}_runmeta.pkl"

    # =========================================================
    # MCMC
    # =========================================================
    def MCMC(self, theta, niters, chain_id):
        Age_Model = np.asarray(theta, float).copy()
        cur_post = self.Log_Posterior(Age_Model)

        if not np.isfinite(cur_post):
            raise ValueError("Initial Age_Model has non-finite posterior.")

        tuning_file = self._tuning_path(chain_id)
        if tuning_file.exists() and self.Start_from_pickles:
            with open(tuning_file, "rb") as f:
                tune = float(pickle.load(f))
        else:
            tune = max(1e-3, np.nanmean(0.5 * (self.low_age_err + self.high_age_err)))

        niters = int(niters)
        burn = int(self.burn_in)
        thin = int(self.thin)

        if niters <= 1 or niters - 1 <= burn:
            n_keep = 0
        else:
            n_keep = 1 + ((niters - 1 - burn) // thin)

        Age_store = np.empty((n_keep, self.resolution), float) if n_keep > 0 else np.empty((0, self.resolution))
        Growth_store = np.empty((n_keep, self.resolution - 1), float) if n_keep > 0 else np.empty((0, self.resolution - 1))
        post_store = np.empty((n_keep,), float) if n_keep > 0 else np.empty((0,))

        prop_count = 0
        acc_count = 0
        save_idx = 0

        pbar = tqdm(
            range(1, niters),
            desc=f"Chain {chain_id}",
            total=niters - 1,
            leave=False,
            dynamic_ncols=True,
            disable=(chain_id != 0) or sys.stdout.isatty(),
        )

        for i in pbar:
            prop = self.propose_state(Age_Model, tune)
            prop_post = self.Log_Posterior(prop)

            accept = False
            if np.isfinite(prop_post):
                log_alpha = prop_post - cur_post
                if (log_alpha >= 0.0) or (np.log(np.random.rand()) < log_alpha):
                    Age_Model = prop
                    cur_post = prop_post
                    accept = True

            prop_count += 1
            if accept:
                acc_count += 1

            if (i < burn) and (i % 200 == 0):
                acc_rate = acc_count / max(prop_count, 1)
                if acc_rate < 0.15:
                    tune *= 0.8
                elif acc_rate > 0.45:
                    tune *= 1.2
                tune = float(np.clip(tune, 1e-4, max(1.0, self.age_ceiling)))
                prop_count = 0
                acc_count = 0

            if (i >= burn) and (((i - burn) % thin) == 0) and (save_idx < n_keep):
                Age_store[save_idx, :] = Age_Model
                Growth_store[save_idx, :] = np.diff(Age_Model) / np.maximum(np.diff(self.depth_grid), 1e-12)
                post_store[save_idx] = cur_post
                save_idx += 1

        with open(self._theta_path(chain_id), "wb") as f:
            pickle.dump(Age_Model, f)
        with open(self._tuning_path(chain_id), "wb") as f:
            pickle.dump(tune, f)

        return Age_store, Growth_store, post_store

    def check_starting_parameters(self):
        starts = []
        for chain_id in range(self.n_chains):
            path = self._theta_path(chain_id)
            if self.Start_from_pickles and path.exists():
                with open(path, "rb") as f:
                    starts.append(pickle.load(f))
            else:
                starts.append(None)

        if any(s is None for s in starts):
            starts = self.Initial_Guesses_for_Model()
            print("Starting from fresh initial guesses.")
        else:
            print(f"Loaded {self.n_chains} chains from pickles.")
        return starts

    def Run_MCMC_Strat(self):
        starts = self.check_starting_parameters()
        total_iterations = int(self.iterations + self.burn_in)

        def run_chain(theta, chain_id):
            return self.MCMC(theta, total_iterations, chain_id)

        results = Parallel(n_jobs=-1)(
            delayed(run_chain)(theta, chain_id)
            for theta, chain_id in zip(starts[:self.n_chains], range(self.n_chains))
        )

        self.Chain_Results = results

        meta = {
            "sample_name": self.sample_name,
            "iterations": self.iterations,
            "burn_in": self.burn_in,
            "thin": self.thin,
            "resolution": self.resolution,
            "pad_frac": self.pad_frac,
            "sigma_extra": self.sigma_extra,
            "smoothness": self.smoothness,
            "n_chains": self.n_chains,
            "model_top_depth": self.model_top_depth,
            "model_bottom_depth": self.model_bottom_depth,
        }
        with open(self._meta_path(), "wb") as f:
            pickle.dump(meta, f)

        return self.Chain_Results

    # =========================================================
    # Post-processing
    # =========================================================
    def _stack_draws(self, which="age"):
        if self.Chain_Results is None:
            self.Run_MCMC_Strat()

        arrs = []
        for age_store, growth_store, post_store in self.Chain_Results:
            if which == "age":
                if age_store.shape[0] > 0:
                    arrs.append(age_store)
            elif which == "growth":
                if growth_store.shape[0] > 0:
                    arrs.append(growth_store)
            else:
                raise ValueError("which must be 'age' or 'growth'")

        if len(arrs) == 0:
            raise ValueError(f"No posterior draws found for '{which}'.")
        return np.concatenate(arrs, axis=0)

    def Get_Age_Model(self):
        draws = self._stack_draws("age")
        low, med, high = self._safe_percentiles(draws, qs=(2.5, 50, 97.5))
        return low, high, med

    def Get_Growth_Model(self):
        draws = self._stack_draws("growth")
        low, med, high = self._safe_percentiles(draws, qs=(2.5, 50, 97.5))
        return low, high, med

    def Get_Ages_At_Depths(self, depth_query, return_full=False,
                           bounds_error=False, fill_value=np.nan):
        scalar_input = np.isscalar(depth_query)
        depth_query = np.atleast_1d(np.asarray(depth_query, dtype=float))

        age_draws = self._stack_draws("age")
        out = np.full((age_draws.shape[0], depth_query.size), np.nan)

        for j in range(age_draws.shape[0]):
            f = interp1d(
                self.depth_grid,
                age_draws[j],
                kind="linear",
                bounds_error=bounds_error,
                fill_value=fill_value,
                assume_sorted=True,
            )
            out[j, :] = f(depth_query)

        low, med, high = self._safe_percentiles(out, qs=(2.5, 50, 97.5))

        if scalar_input:
            if return_full:
                return low[0], high[0], med[0], out[:, 0]
            return low[0], high[0], med[0]

        if return_full:
            return low, high, med, out
        return low, high, med

    def Get_Age_Depth_Plot(self):
        low_age, high_age, age_med = self.Get_Age_Model()
        err_asymm = [self.low_age_err * 2.0, self.high_age_err * 2.0]

        fig, ax = plt.subplots(figsize=(5, 7))
        ax.errorbar(
            x=self.u_ages,
            y=self.Depths,
            xerr=err_asymm,
            fmt="o",
            markerfacecolor="dodgerblue",
            markeredgecolor="k",
            ecolor="k",
            capsize=4,
            label="U-series ages (2σ)",
        )
        ax.fill_betweenx(self.depth_grid, low_age, high_age, alpha=0.3, label="95% CI")
        ax.plot(age_med, self.depth_grid, lw=1.5, label="Median model")
        ax.invert_yaxis()
        ax.set_xlabel("Age")
        ax.set_ylabel("Depth")
        ax.legend()
        return fig, ax

    def Save_Age_Depth_Model(self):
        low_age, high_age, age_median = self.Get_Age_Model()

        df_age_depth = pd.DataFrame(
            {
                "Depth": self.depth_grid,
                "Age_Median": age_median,
                "Age_low_95": low_age,
                "Age_high_95": high_age,
            }
        )

        output_path = self.save_dir / f"{self.sample_name}_Age_Depth_Model.csv"
        df_age_depth.to_csv(output_path, index=False)
        print(f"Saved Age-Depth model to: {output_path}")
