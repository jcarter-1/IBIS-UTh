"""IBIS 0.3.2 Part 1 web interface: Thoth prior and sample-specific U–Th ages.

Launch with: uvicorn app:app --reload
"""
from __future__ import annotations

import base64
import pickle
import io
import hashlib
import json
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from ibis.IBIS_Main import IBIS
from ibis.IBIS_Thoth_V2 import U_Series_Age_Equation


BASE = Path(__file__).resolve().parent
NAVY, BLUE, RED, GOLD, GREY = "#213555", "#7C9AB7", "#BA5C58", "#C89D57", "#8997A8"

app = FastAPI(title="IBIS | Initial Thorium & U–Th Ages", version="0.3.2-web")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ibis")
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


class Analysis(BaseModel):
    Depth: float
    U234_r: float = Field(gt=0)
    U234_r_err: float = Field(gt=0)
    Th230_r: float = Field(gt=0)
    Th230_r_err: float = Field(gt=0)
    Th232_r: float = Field(gt=0)
    Th232_r_err: float = Field(gt=0)
    Depth_err: float | None = Field(default=None, ge=0)


class InitialThReference(BaseModel):
    sample: str = Field(min_length=1, max_length=80)
    depth: float
    initial_th: float = Field(ge=0)
    sigma: float = Field(gt=0)


class Settings(BaseModel):
    fraction_det: float = Field(default=0.5, ge=0, le=1)
    global_mass: float = Field(default=0.8, ge=0, le=1)
    use_r02_in_sigma: bool = False
    n_samples_global: int = Field(default=500, ge=100, le=20000)
    n_samples_layer: int = Field(default=150, ge=50, le=20000)
    neighbors: int = Field(default=1, ge=0, le=5)
    n_chains: int = Field(default=3, ge=2, le=6)
    iterations: int = Field(default=4000, ge=100, le=2000000)
    burn_in: int = Field(default=2000, ge=0, le=2000000)


class PriorRequest(BaseModel):
    rows: list[Analysis]
    settings: Settings = Field(default_factory=Settings)
    reference: list[InitialThReference] | None = None

    @model_validator(mode="after")
    def validate_data(self):
        if not 3 <= len(self.rows) <= 200:
            raise ValueError("Supply 3–200 measurements for IBIS Part 1.")
        data = pd.DataFrame([row.model_dump() for row in self.rows])
        if not np.isfinite(data.drop(columns=["Depth_err"]).to_numpy(dtype=float)).all():
            raise ValueError("All ratios, 1σ uncertainties, and depths must be finite.")
        if not np.isfinite(data["Depth_err"].dropna().to_numpy(dtype=float)).all():
            raise ValueError("Depth uncertainties must be finite.")
        if data["Depth"].nunique() < 2:
            raise ValueError("At least two distinct depths are needed.")
        if self.reference is not None:
            depths = [r.depth for r in self.reference]
            if (len(depths) != len(self.rows) or len(set(depths)) != len(depths)
                    or sorted(depths) != sorted(data["Depth"].tolist())):
                raise ValueError("Reference points must match the measurements one-to-one by unique depth.")
        return self


class MCMCSettings(BaseModel):
    n_chains: int = Field(default=3, ge=2, le=6)
    iterations: int = Field(default=4000, ge=100, le=2000000)
    burn_in: int = Field(default=2000, ge=0, le=2000000)


class Part1Request(BaseModel):
    prior_job_id: str
    mcmc: MCMCSettings = Field(default_factory=MCMCSettings)


_COLUMN_ALIASES = {
    "depth": "Depth", "depths": "Depth", "depthtop": "Depth",
    "deptherr": "Depth_err", "depthserr": "Depth_err",
    "u234r": "U234_r", "u234238actratio": "U234_r", "u234238ratios": "U234_r",
    "u234rerr": "U234_r_err", "u234238actratiouncertainty": "U234_r_err",
    "u234238ratioserr": "U234_r_err", "u234238actratiouncert": "U234_r_err",
    "th230r": "Th230_r", "th230238actratio": "Th230_r", "th230238uratios": "Th230_r",
    "th230rerr": "Th230_r_err", "th230238actratiouncertainty": "Th230_r_err",
    "th230238uratioserr": "Th230_r_err", "th230238actratiouncert": "Th230_r_err",
    "th232r": "Th232_r", "th232238actratio": "Th232_r", "th232238uratios": "Th232_r",
    "th232rerr": "Th232_r_err", "th232238actratiouncertainty": "Th232_r_err",
    "th232238uratioserr": "Th232_r_err", "th232238actratiouncert": "Th232_r_err",
}


def _normalize_upload(data: pd.DataFrame) -> list[dict]:
    """Accept Faraji's original workbook headings and canonical IBIS headings."""
    renamed = {}
    for col in data.columns:
        key = "".join(c for c in str(col).lower() if c.isalnum())
        target = _COLUMN_ALIASES.get(key)
        if target is not None:
            if target in renamed.values():
                raise ValueError(f"Multiple columns map to {target}; rename the duplicates before uploading.")
            renamed[col] = target

    df = data.rename(columns=renamed)
    required = ["Depth", "U234_r", "U234_r_err", "Th230_r", "Th230_r_err", "Th232_r", "Th232_r_err"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    available = required + (["Depth_err"] if "Depth_err" in df.columns else [])
    df = df[available].copy()
    df = df.loc[~df[required].isna().all(axis=1)].reset_index(drop=True)
    if df.empty or len(df) > 200:
        raise ValueError("Upload 1–200 measurements per file.")
    for name in available:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    if not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("A required measurement is blank or not numeric.")
    if (df[required[1:]] <= 0).any().any():
        raise ValueError("Ratios and absolute 1σ uncertainties must be positive.")
    if "Depth_err" in df.columns:
        invalid = df["Depth_err"].notna() & (~np.isfinite(df["Depth_err"]) | (df["Depth_err"] < 0))
        if invalid.any():
            raise ValueError("Depth uncertainties must be finite and nonnegative.")
        df["Depth_err"] = df["Depth_err"].astype(object).where(df["Depth_err"].notna(), None)
    else:
        df["Depth_err"] = None
    return df.to_dict(orient="records")


def _set_job(job_id: str, **fields):
    with _lock:
        _jobs[job_id].update(fields)


def _start(kind: Literal["prior", "part1"], source=None):
    with _lock:
        if len(_jobs) >= 30:
            raise HTTPException(503, "Too many jobs in this session; restart the local server.")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "stage": "Waiting to start", "kind": kind}
    fn = _build_prior if kind == "prior" else _run_part1
    _executor.submit(_run_safe, fn, job_id, source)
    return {"job_id": job_id}


def _run_safe(fn, job_id, source):
    try:
        _set_job(job_id, status="running")
        fn(job_id, source)
    except Exception as exc:
        _set_job(job_id, status="error", stage="Stopped", error=str(exc))


def _make_input(request: PriorRequest, folder: Path) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump() for row in request.rows])
    df = df.sort_values("Depth", kind="stable").reset_index(drop=True)
    # User-provided values are absolute 1σ activity-ratio uncertainties.
    # A missing depth uncertainty is unimportant for Part 1, but IBIS expects it.
    df["Depth_err"] = pd.to_numeric(df["Depth_err"], errors="raise").fillna(0.0)
    df.to_csv(folder / "measurements.csv", index=False)
    return df


def _meta(settings: Settings) -> dict:
    # IBIS 0.3.2's fraction_det key is currently not read by the Thoth sampler.
    # fraction_bulk is the ACTUAL parameter used in boutique_thoriums().
    return {
        "fraction_det": settings.fraction_det,
        "fraction_bulk": 1.0 - settings.fraction_det,
        "global_mass": settings.global_mass,
        "use_r02_in_sigma": settings.use_r02_in_sigma,
        "n_samples_global": settings.n_samples_global,
        "n_samples_layer": settings.n_samples_layer,
        "neighbors": settings.neighbors,
        "candidate_factor": 3,
        "batch_size": 300,
        "cv": 3,
        "n_jobs": 1,
        "print_diagnostics": False,
    }


def _model(folder: Path, settings: Settings, *, start_from_pickles=False) -> IBIS:
    return IBIS(
        filepath=folder / "measurements.csv",
        sample_name="IBIS_Web",
        MCMC_samples=settings.iterations,
        MCMC_burn_in=settings.burn_in,
        n_chains=settings.n_chains,
        Start_from_pickles=start_from_pickles,
        Top_Age_Stal=False,
        show_bird=False,
        method="thoth",
        diction_meta=_meta(settings),
        save_dir=folder,
        n_jobs=1,
    )


def _png(fig) -> str:
    out = BytesIO()
    fig.savefig(out, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _plot_prior(prior):
    x = np.asarray(prior.grid, dtype=float)
    y = np.asarray(prior.pdf_grid, dtype=float)
    areas = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    cdf = np.r_[0, np.cumsum(areas)]
    cdf /= cdf[-1]
    q = lambda p: float(np.interp(p, cdf, x))
    lo, median, hi = q(0.025), q(0.5), q(0.975)
    xmax = max(0.83, q(0.998)) * 1.08

    fig, ax = plt.subplots(figsize=(9.3, 4.2))
    ax.fill_between(x, y, where=(x >= lo) & (x <= hi), color=BLUE, alpha=0.34, label="95% prior interval")
    ax.plot(x, y, color=NAVY, lw=2.4, label="Thoth prior")
    ax.axvline(median, color=NAVY, lw=1.2, ls=":", label=f"Median {median:.3f}")
    ax.axvline(0.83, color=RED, lw=1.5, ls="--", label="Bulk Earth: 0.83")
    ax.set(xlim=(0, xmax), ylim=(0, None), xlabel=r"Initial ($^{230}$Th/$^{232}$Th)$_A$", ylabel="Probability density")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    _axes(ax)
    fig.tight_layout()
    return _png(fig), {"median": median, "lo95": lo, "hi95": hi}


def _axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#CBD3DD")
    ax.spines["bottom"].set_color("#CBD3DD")
    ax.tick_params(colors="#526477", labelsize=10)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.grid(axis="x", color="#E5EBF2", lw=0.7)
    ax.set_axisbelow(True)


def _comparisons(df: pd.DataFrame):
    result = []
    for _, r in df.iterrows():
        item = {"Depth": float(r["Depth"])}
        for key, th0, th0_err in (("uncorrected", 0.0, 0.0), ("bulk", 0.83, 0.43)):
            eq = U_Series_Age_Equation(
                r08=r["Th230_r"], r08_err=r["Th230_r_err"],
                r28=r["Th232_r"], r28_err=r["Th232_r_err"],
                r48=r["U234_r"], r48_err=r["U234_r_err"],
                r02_initial=th0, r02_initial_err=th0_err,
            )
            age, sigma = eq.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
            item[f"{key}_age"] = float(age)
            item[f"{key}_sigma"] = float(sigma)
        result.append(item)
    if not np.isfinite(pd.DataFrame(result).to_numpy(dtype=float)).all():
        raise ValueError("Unable to compute finite comparison ages for every analysis.")
    return result


def _build_prior(job_id, request: PriorRequest):
    tmp = tempfile.TemporaryDirectory(prefix="ibis-part1-")
    folder = Path(tmp.name)
    _set_job(job_id, stage="Preparing U–Th measurements")
    df = _make_input(request, folder)
    comparison = _comparisons(df)
    _set_job(job_id, stage="Sampling stratigraphy-informed Thoth prior")
    model = _model(folder, request.settings)
    prior = model.Set_Up_MCMC()
    _set_job(job_id, stage="Drawing initial-thorium prior")
    plot, summary = _plot_prior(prior)
    references = None
    if request.reference is not None:
        references = [r.model_dump() for r in sorted(request.reference, key=lambda r: r.depth)]
    _set_job(
        job_id, status="complete", stage="Thoth prior ready",
        result={"prior_plot": plot, "prior_summary": summary, "comparisons": comparison,
                "settings": request.settings.model_dump(), "references": references},
        temp_dir=tmp, folder=str(folder),
    )


def _plot_age_comparison(rows: list[dict]):
    depths = np.array([r["Depth"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(9.1, max(4.8, 0.35 * len(rows) + 2)))
    for col, label, color, offset in (
        ("uncorrected", "Uncorrected", GREY, -0.18),
        ("bulk", "Bulk Earth", GOLD, 0.00),
        ("ibis", "IBIS Part 1", RED, 0.18),
    ):
        ages = np.array([r[f"{col}_age"] for r in rows])
        if col == "ibis":
            # Use the actual asymmetric posterior 95% interval for IBIS.
            xerr = np.vstack((ages - [r["ibis_lo95"] for r in rows],
                              [r["ibis_hi95"] for r in rows] - ages))
        else:
            # Traditional analytical errors: ±2 propagated standard deviations.
            xerr = 2 * np.array([r[f"{col}_sigma"] for r in rows])
        # Horizontal age uncertainties; use sample ranks so very close depths do not overlap.
        ordinal = np.arange(len(rows)) + offset
        ax.errorbar(ages, ordinal, xerr=xerr, fmt="o", color=color, ecolor=color,
                    capsize=2, markersize=5.2, lw=1.35, alpha=0.95, label=label)
    ax.set_yticks(np.arange(len(depths)), [f"{d:g}" for d in depths])
    ax.invert_yaxis()
    ax.set(xlabel="Age (years before measurement)", ylabel="Depth (input units)")
    ax.legend(frameon=False, loc="best", ncol=3, fontsize=9)
    ax.set_title("U–Th ages by analysis  ·  approximately 95% intervals", loc="left", fontsize=12, color=NAVY, pad=17)
    _axes(ax)
    fig.tight_layout()
    return _png(fig)


def _plot_thorium(rows: list[dict]):
    depths = np.array([r["Depth"] for r in rows])
    med = np.array([r["th0"] for r in rows])
    lo = np.array([r["th0_lo95"] for r in rows])
    hi = np.array([r["th0_hi95"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.32 * len(rows) + 2)))
    ax.errorbar(med, depths, xerr=np.vstack([med - lo, hi - med]), fmt="o", color=NAVY,
                markerfacecolor=BLUE, markeredgecolor=NAVY, capsize=3, lw=1.5, label="IBIS median and 95% interval")
    ax.axvspan(0, 0.83 + 2 * 0.43, color=GOLD, alpha=0.10)
    ax.axvline(0.83, color=GOLD, ls="--", lw=1.4, label="Bulk Earth")
    ax.invert_yaxis()
    ax.set(xlabel=r"Initial ($^{230}$Th/$^{232}$Th)$_A$", ylabel="Depth (input units)")
    ax.set_title("Sample-specific initial thorium", loc="left", fontsize=12, color=NAVY, pad=17)
    ax.legend(frameon=False, fontsize=9)
    _axes(ax)
    fig.tight_layout()
    return _png(fig)


def _plot_faraji_reference(rows: list[dict]):
    """Compare IBIS Part 1 initial-Th posterior to laminae back-calculation."""
    x = np.array([r["faraji_initial"] for r in rows])
    sx = np.array([r["faraji_sigma"] for r in rows])
    y = np.array([r["th0"] for r in rows])
    sy = np.array([r["th0_sigma"] for r in rows])
    depths = np.array([r["Depth"] for r in rows])

    lo = max(0.0, float(min(np.min(x - 2 * sx), np.min(y - 2 * sy))))
    hi = float(max(np.max(x + 2 * sx), np.max(y + 2 * sy)))
    pad = max(0.2, (hi - lo) * 0.06)

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=RED,
            ls="--", lw=1.6, alpha=0.9, zorder=1, label="1:1 agreement")
    ax.errorbar(x, y, xerr=sx, yerr=sy, ls="none", color="#7E8FA3",
                elinewidth=1.3, capsize=2.8, alpha=0.8, zorder=2)
    points = ax.scatter(x, y, c=depths, cmap="cividis", s=73,
                        edgecolor="white", linewidth=1.1, zorder=3)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
           xlabel=r"Faraji et al. (2021) back-calculated ($^{230}$Th/$^{232}$Th)$_{A0}$",
           ylabel=r"IBIS posterior ($^{230}$Th/$^{232}$Th)$_{A0}$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Initial thorium: IBIS versus laminae chronology", loc="left",
                 fontsize=12, color=NAVY, pad=16)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    cbar = fig.colorbar(points, ax=ax, fraction=0.045, pad=0.05)
    cbar.set_label("Depth (input units)", color=NAVY)
    cbar.outline.set_visible(False)
    _axes(ax)
    fig.tight_layout()
    return _png(fig)


def _split_chains(draws):
    """Split each chain in half for nonstationarity-sensitive diagnostics."""
    x = np.asarray(draws, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a 2D array: (chain, retained_draw).")
    m, n = x.shape
    if m < 2 or n < 8 or not np.all(np.isfinite(x)):
        return None
    half = n // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def _split_rhat(draws):
    """Classical split Gelman-Rubin R-hat (not rank-normalized)."""
    x = _split_chains(draws)
    if x is None:
        return None
    m, n = x.shape
    w = float(np.var(x, axis=1, ddof=1).mean())
    b = float(n * np.var(x.mean(axis=1), ddof=1))
    if w <= 0:
        return None  # constant chains: R-hat undefined
    value = np.sqrt(((n - 1) / n * w + b / n) / w)
    return float(value) if np.isfinite(value) else None


def _autocovariance_fft(x):
    """Autocovariance of rows of x, normalized by N (FFT method)."""
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft, axis=1)
    return np.fft.irfft(f * np.conjugate(f), n=nfft, axis=1)[:, :n] / n


def _ess_autocorrelation(draws):
    """Approximate split-chain autocorrelation ESS using positive pairs.

    This is not rank-normalized bulk ESS or quantile/tail ESS.
    """
    x = _split_chains(draws)
    if x is None:
        return None
    m, n = x.shape
    within = float(np.var(x, axis=1, ddof=1).mean())
    between = float(n * np.var(x.mean(axis=1), ddof=1))
    var_plus = (n - 1) / n * within + between / n
    if not np.isfinite(var_plus) or var_plus <= 0:
        return None
    acov = _autocovariance_fft(x).mean(axis=0)
    # Split-chain autocorrelation estimator with between-chain correction.
    rho = 1.0 - (within - acov) / var_plus
    rho[0] = 1.0
    # Initial positive sequence of adjacent autocorrelation pairs.
    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = float(rho[k] + rho[k + 1])
        if not np.isfinite(pair) or pair <= 0:
            break
        tau += 2.0 * pair
    tau = max(1.0, tau)
    value = (m * n) / tau
    return float(np.clip(value, 1.0, m * n)) if np.isfinite(value) else None


def _extract_chain_diagnostics(model):
    """Use the actual stored IBIS_MCMC Chain_Results arrays (no ArviZ)."""
    chains = getattr(model, "Ibis_Chains", None)
    raw = getattr(chains, "Chain_Results", None)
    if raw is None or len(raw) < 2:
        return {"status": "unavailable", "message": "No chain-separated samples found."}, None

    names = ("Age", "Initial Th", "234U/238U", "232Th/238U", "230Th/238U", "Initial 234U/238U")
    indexes = (0, 1, 2, 3, 4, 6)
    try:
        arrays = {name: np.stack([np.asarray(chain[idx], dtype=float) for chain in raw])
                  for name, idx in zip(names, indexes)}
        arrays["Log posterior"] = np.stack([np.asarray(chain[5], dtype=float) for chain in raw])
        c, n = arrays["Log posterior"].shape
        if c < 2 or n < 8:
            return {"status": "unavailable", "message": "Need >=2 chains and >=8 retained draws."}, arrays
        if not all(np.isfinite(a).all() for a in arrays.values()):
            return {"status": "unavailable", "message": "Non-finite values found in chain output."}, arrays
        stats = []
        for name in names:
            a = arrays[name]
            for j in range(a.shape[2]):
                sample = a[:, :, j]
                stats.append({
                    "parameter": name,
                    "depth_index": j,
                    "rhat": _split_rhat(sample),
                    "ess_bulk": _ess_autocorrelation(sample),  # legacy API key for frontend
                    "ess_tail": None,
                })
        good_r = [r["rhat"] for r in stats if r["rhat"] is not None]
        good_ess = [r["ess_bulk"] for r in stats if r["ess_bulk"] is not None]
        return {
            "status": "available",
            "message": "Classical split R-hat and approximate autocorrelation ESS from retained draws (NumPy only). Tail ESS not calculated.",
            "n_chains": c, "n_draws": n, "parameters": list(names), "stats": stats,
            "max_rhat": max(good_r) if good_r else None,
            "min_ess_bulk": min(good_ess) if good_ess else None,
        }, arrays
    except (ValueError, TypeError, IndexError) as exc:
        return {"status": "unavailable", "message": f"Cannot read chain arrays: {exc}"}, None


def _run_part1(job_id, run_request: Part1Request):
    prior_job_id = run_request.prior_job_id
    with _lock:
        source = _jobs.get(prior_job_id)
        if not source or source.get("kind") != "prior" or source.get("status") != "complete":
            raise ValueError("The Thoth prior is missing or has not completed.")
        folder = Path(source["folder"])
        settings = Settings(**source["result"]["settings"])
        # MCMC controls are independent from the cached Thoth configuration.
        settings = settings.model_copy(update=run_request.mcmc.model_dump())
        comparison = [dict(r) for r in source["result"]["comparisons"]]
        references = source["result"].get("references")
    _set_job(job_id, stage=f"Running IBIS Part 1 · {settings.n_chains} MCMC chains")
    model = _model(folder, settings, start_from_pickles=bool(source.get("imported")))
    model.Run_MCMC()  # Explicitly no Run_MCMC_Strat() / age-depth modeling.
    _set_job(job_id, stage="Summarizing posterior ages and initial thorium")
    posterior = model.Ibis_Chains.MakeSummaryDataFrame().sort_values("Depth_Meas", kind="stable").reset_index(drop=True)
    if len(posterior) != len(comparison):
        raise ValueError("IBIS posterior length does not match the input analyses.")

    # ---------------------------------------------------------
    # Read the original U-Th measurements
    # ---------------------------------------------------------

    measurements = (
        pd.read_csv(folder / "measurements.csv")
        .sort_values("Depth", kind="stable")
        .reset_index(drop=True)
    )

    if len(measurements) != len(posterior):
        raise ValueError(
            "Measurement and posterior row counts do not match."
        )
    rows = []
    for i, row in posterior.iterrows():
        r = dict(comparison[i])
        # Include the original analytical measurements
        measurement = measurements.iloc[i]

        # Verify measurements are aligned with posterior samples
        if not np.isclose(
            measurement["Depth"],
            row["Depth_Meas"],
            atol=1e-9,
            rtol=1e-9
        ):
            raise ValueError("Measurement/posterior depth mismatch.")

        for column in [
            "U234_r",
            "U234_r_err",
            "Th230_r",
            "Th230_r_err",
            "Th232_r",
            "Th232_r_err"
        ]:
            r[column] = float(measurement[column])

        # Depth uncertainty is optional
        depth_err = measurement["Depth_err"]

        r["Depth_err"] = (
            float(depth_err)
            if pd.notna(depth_err)
            else None
            )   
        if not np.isclose(r["Depth"], row["Depth_Meas"], atol=1e-9, rtol=1e-9):
            raise ValueError("IBIS sample order mismatch; results were not assigned to the wrong depths.")
        r.update(
            ibis_age=float(row["age"]),
            ibis_sigma=float(row["age_err_sym"]),
            ibis_lo68=float(row["age_lo68"]), ibis_hi68=float(row["age_hi68"]),
            ibis_lo95=float(row["age_lo95"]), ibis_hi95=float(row["age_hi95"]),
            th0=float(row["initial thorium"]), th0_sigma=float(row["initial thorium err"]),
            th0_lo95=float(row["Th0_lo95"]), th0_hi95=float(row["Th0_hi95"]),
            u0=float(row["initial uranium"]), u0_sigma=float(row["initial uranium err"]),
        )
        if references is not None:
            ref = references[i]
            if not np.isclose(ref["depth"], r["Depth"], atol=1e-9, rtol=1e-9):
                raise ValueError("Reference depth does not match the IBIS posterior sample.")
            r.update(sample=ref["sample"], faraji_initial=ref["initial_th"],
                     faraji_sigma=ref["sigma"], th0_difference=float(row["initial thorium"]) - ref["initial_th"])
        rows.append(r)
    numeric_rows = pd.DataFrame(rows).select_dtypes(include="number")
    if not np.isfinite(numeric_rows.to_numpy(dtype=float)).all():
        raise ValueError("IBIS returned a non-finite posterior summary.")
    _set_job(job_id, stage="Drawing comparison and posterior figures")
    diagnostics, trace_arrays = _extract_chain_diagnostics(model)
    result = {
        "rows": rows,
        "prior_job_id": prior_job_id,
        "mcmc_settings_used": run_request.mcmc.model_dump(),
        "diagnostics": diagnostics,
    }
    if references is not None:
        result["faraji_plot"] = _plot_faraji_reference(rows)
        result["faraji_summary"] = {
            "n": len(rows),
            "mean_difference": float(np.mean([r["th0_difference"] for r in rows])),
            "rmse": float(np.sqrt(np.mean([r["th0_difference"] ** 2 for r in rows]))),
        }
    _set_job(job_id, status="complete", stage="IBIS Part 1 complete", result=result, trace_arrays=trace_arrays)


# ============================================================
# DATA EXPLORATION — CONVENTIONAL U–Th AGES
# ============================================================

class ConventionalAgeRequest(BaseModel):
    rows: list[Analysis]


# A prior workspace bundle stores the actual files produced by Set_Up_MCMC,
# not just its plotted PDF. Only upload files you created/trust: cached IBIS
# .pkl files may be deserialized by the installed IBIS package.
PRIOR_FORMAT = "ibis-thoth-workspace-v1"
MAX_BUNDLE_SIZE = 30_000_000
MAX_FILE_BYTES = 15_000_000


class _RestrictedBundleUnpickler(pickle.Unpickler):
    """The OUTER .pkl must contain only basic Python data containers."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Executable pickle classes are forbidden")


def _prior_bundle(job: dict) -> bytes:
    folder = Path(job["folder"])
    files = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if len(rel.parts) > 4 or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Prior workspace contains an unsupported file")
        files[rel.as_posix()] = path.read_bytes()
    if "measurements.csv" not in files:
        raise ValueError("Prior workspace is missing measurements.csv")
    obj = {
        "format": PRIOR_FORMAT,
        "settings": job["result"]["settings"],
        "comparisons": job["result"]["comparisons"],
        "references": job["result"].get("references"),
        "prior_summary": job["result"]["prior_summary"],
        "prior_plot": job["result"]["prior_plot"],
        "files": files,
    }
    output = pickle.dumps(obj, protocol=4)
    if len(output) > MAX_BUNDLE_SIZE:
        raise ValueError("Prior bundle exceeds the 30 MB limit")
    return output


@app.get("/api/prior/{job_id}/download")
def download_prior(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("kind") != "prior" or job.get("status") != "complete":
            raise HTTPException(404, "Completed prior not found")
    try:
        payload = _prior_bundle(job)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(payload, media_type="application/octet-stream", headers={
        "Content-Disposition": 'attachment; filename="IBIS_Thoth_Prior.pkl"',
        "Cache-Control": "no-store",
    })


@app.post("/api/prior/upload")
async def upload_prior(request: Request):
    body = await request.body()
    if not body or len(body) > MAX_BUNDLE_SIZE:
        raise HTTPException(413, "Select a saved IBIS Thoth .pkl smaller than 30 MB")
    try:
        obj = _RestrictedBundleUnpickler(io.BytesIO(body)).load()
        if not isinstance(obj, dict) or obj.get("format") != PRIOR_FORMAT:
            raise ValueError("Not an IBIS Thoth workspace bundle")
        files = obj["files"]
        if not isinstance(files, dict) or "measurements.csv" not in files or len(files) > 500:
            raise ValueError("Invalid prior workspace")
        settings = Settings(**obj["settings"])
        comparison = obj["comparisons"]
        summary = obj["prior_summary"]
        plot = obj["prior_plot"]
        references = obj.get("references")
        if not isinstance(comparison, list) or not isinstance(summary, dict) or not isinstance(plot, str):
            raise ValueError("Invalid prior summary")
        tmp = tempfile.TemporaryDirectory(prefix="ibis-restored-prior-")
        folder = Path(tmp.name)
        for name, contents in files.items():
            rel = Path(name)
            if (not isinstance(name, str) or not isinstance(contents, bytes)
                    or len(rel.parts) > 4 or rel.is_absolute()
                    or ".." in rel.parts or len(contents) > MAX_FILE_BYTES):
                raise ValueError("Invalid file in prior bundle")
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        df = pd.read_csv(folder / "measurements.csv")
        if not 3 <= len(df) <= 200 or len(df) != len(comparison):
            raise ValueError("Inconsistent saved measurements")
        # The browser compares this fingerprint against its current table.
        saved_rows = df.replace({np.nan: None}).to_dict(orient="records")
        with _lock:
            if len(_jobs) >= 30:
                raise ValueError("Too many jobs; restart the server")
            job_id = uuid.uuid4().hex
            _jobs[job_id] = {
                "status": "complete", "kind": "prior", "stage": "Prior restored",
                "imported": True, "folder": str(folder), "temp_dir": tmp,
                "result": {
                    "prior_plot": plot, "prior_summary": summary,
                    "comparisons": comparison,
                    "settings": settings.model_dump(),
                    "references": references,
                }
            }
        return {"job_id": job_id, "result": _jobs[job_id]["result"],
                "saved_rows": saved_rows}
    except (ValueError, KeyError, TypeError, pickle.UnpicklingError) as exc:
        raise HTTPException(400, f"Cannot restore prior: {exc}") from exc


@app.post("/api/conventional-ages")
def conventional_ages(request: ConventionalAgeRequest):

    # Convert the submitted measurement rows to a DataFrame
    df = pd.DataFrame([
        row.model_dump() for row in request.rows
    ])

    # Sort by stratigraphic depth
    df = df.sort_values(
        "Depth", kind="stable"
    ).reset_index(drop=True)

    # Reuse the existing IBIS age-equation calculations
    results = _comparisons(df)

    return {"rows": results}


@app.post("/api/prior", status_code=202)
def prior(request: PriorRequest):
    return _start("prior", request)


@app.post("/api/import")
async def import_measurements(request: Request):
    """Parse Excel or CSV in Python so the browser needs no spreadsheet library."""
    filename = Path(request.query_params.get("filename", "")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".csv"}:
        raise HTTPException(400, "Choose an Excel .xlsx or CSV .csv file.")
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 5_000_000:
            raise HTTPException(413, "Maximum file size is 5 MB.")
        chunks.append(chunk)
    if not size:
        raise HTTPException(400, "The selected file is empty.")
    try:
        stream = BytesIO(b"".join(chunks))
        df = pd.read_excel(stream, sheet_name=0) if suffix == ".xlsx" else pd.read_csv(stream)
        rows = _normalize_upload(df)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, "Could not read this file. Check the format and column names.") from exc
    return {"rows": rows, "filename": filename}


@app.post("/api/part1", status_code=202)
def part1(request: Part1Request):
    with _lock:
        prior = _jobs.get(request.prior_job_id)
        if not prior or prior["kind"] != "prior" or prior["status"] != "complete":
            raise HTTPException(400, "Generate a successful Thoth prior first.")
    return _start("part1", request)


@app.get("/api/diagnostics/{job_id}/trace")
def diagnostics_trace(job_id: str, parameter: str = "Log posterior", depth_index: int = 0):
    with _lock:
        job = _jobs.get(job_id)
        arrays = job.get("trace_arrays") if job else None
    if arrays is None:
        raise HTTPException(404, "Chain traces are unavailable for this job.")
    if parameter not in arrays:
        raise HTTPException(400, "Unknown parameter")
    arr = arrays[parameter]
    if arr.ndim == 3:
        if not 0 <= depth_index < arr.shape[2]:
            raise HTTPException(400, "Depth index out of range")
        arr = arr[:, :, depth_index]
    step = max(1, int(np.ceil(arr.shape[1] / 1800)))
    draw_index = np.arange(0, arr.shape[1], step)
    return {"parameter": parameter, "depth_index": depth_index,
            "draw_index": draw_index.tolist(), "chains": arr[:, ::step].tolist(), "total_draws": int(arr.shape[1])}


@app.get("/api/jobs/{job_id}")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job ID")
        # Never expose local temp paths or non-JSON serializable objects.
        return {key: job[key] for key in ("status", "stage", "error", "result") if key in job}


@app.get("/api/health")
def health():
    return {"ok": True, "model": "ibis-uth 0.3.2", "stage": "Part 1 only"}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
