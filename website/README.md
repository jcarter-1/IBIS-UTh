# IBIS Part 1 · Web interface

A local FastAPI website for **IBIS U–Th 0.3.2**. It implements only the first stage: an empirical Thoth initial-thorium prior and sample-specific Bayesian corrected ages. **Does not** run `Run_MCMC_Strat()` part of the code for the time being but is a tool for visualizing the the effect of initial thorium correction choices and applying IBIS. 

## Run it on your Mac

1. Install Python

Install Python 3.12, 3.13, or 3.14, which are supported by ibis-uth v0.3.2.

2. Launch the website

Open a terminal, go to the website directory, and run:

bash start_ibis.sh

3. Access IBIS

First launch: The startup script automatically installs the required IBIS dependencies, starts the local server, and opens the website in your browser.

Subsequent launches: The script reuses the existing installation and launches the website directly, without reinstalling dependencies.

The website runs locally on your computer.

## Workflow

1. Enter activity ratios at absolute **1σ** uncertainty and sampling depths, or upload an **Excel `.xlsx` / CSV `.csv`** file with the columns `Depth, U234_r, U234_r_err, Th230_r, Th230_r_err, Th232_r, Th232_r_err`. `Depth_err` is optional. The original Faraji spreadsheet headers (such as `U_234_r` and `Th_230_r_err`) are recognized automatically. The first worksheet is used for Excel files; maximum file size is 5 MB. You can also paste cells directly from Excel into the grid.
2. Set the flexible/detrital proposal fraction, global-versus-local Thoth weight, and the option to propagate *proposed initial-Th uncertainty* into the Thoth age-σ likelihood.
3. Generate the Thoth prior. Once this finishes, click **Run IBIS Part 1** to run MCMC with the same prior and settings.
4. Inspect the prior density, age comparisons, sample-specific initial-Th posterior, and download a CSV with all age estimates and uncertainties.

### Faraji et al. (2021) test case

Click **Faraji et al. (2021) test case** to populate ten analyses from the supplied `Faraji_2021_data.xlsx` and enable the published initial-thorium reference comparison. The test case is shipped as `static/examples/faraji_2021.csv` so it works immediately after installing the website; you can also import the original `.xlsx` yourself as an ordinary dataset.

Run **Generate prior**, then **Run IBIS Part 1**. For the Faraji preset, results include an additional **1:1 diagram** of posterior IBIS initial `(²³⁰Th/²³²Th)ₐ₀` against the ten back-calculated values from laminae ages: `[8.05, 6.00, 5.98, 6.63, 7.74, 6.55, 6.01, 6.21, 6.87, 4.88]`, with published absolute 1σ uncertainties `[0.24, 0.25, 0.27, 0.25, 0.18, 0.21, 0.50, 0.46, 0.27, 0.97]`. Horizontal bars are Faraji uncertainties and vertical bars are IBIS 1σ-equivalent uncertainties; the dashed diagonal represents perfect agreement. Matchups use **depth**. CSV downloads include the matched sample name, Faraji estimate, uncertainty and IBIS-minus-Faraji difference. 

*Citation:* Faraji, M., Borsato, A., Frisia, S., Hellstrom, J. C., Lorrey, A., Hartland, A., Greig, A. & Mattey, D. P. (2021). Accurate dating of stalagmites from low seasonal contrast tropical Pacific climate using Sr 2D maps, fabrics and annual hydrological cycles. *Scientific Reports* **11**, 2178.

## What the settings really mean

| User control | IBIS 0.3.2 parameter | Meaning |
|---|---|---|
| Flexible/detrital proposal fraction | `fraction_bulk = 1 - fraction_det` | Fraction of proposals drawn from a *broad, flexible* initial-Th distribution vs. a Bulk-Earth-like proposal; not a measured detrital mass fraction. `fraction_det` exists in 0.3.2 but its sampler does **not** currently read it, so this mapping is required. |
| Global prior weight | `global_mass` | Mixture weight of the full-record Thoth KDE vs. neighboring-window KDEs. |
| Include proposed Th uncertainty in age σ | `use_r02_in_sigma` | Includes the uncertainty of each proposed initial-Th correction when evaluating the age uncertainties for Thoth stratigraphy. The age-uncertainty terms from U–Th analyses remain included when unchecked. |
| Global / local proposal targets | `n_samples_global`, `n_samples_layer` | Requested valid samples for global/local Thoth prior construction. The website also sets `candidate_factor=3`, `batch_size=300`, and `cv=3` for practical runtime. |
| Chains / iterations / burn-in | `n_chains`, `MCMC_samples`, `MCMC_burn_in` | IBIS Part 1 posterior MCMC. |

The supplied default sampling settings are **exploratory**, not a claim of converged posterior estimates. Increase counts and iterations and examine MCMC convergence before using results for publication.

## Output conventions

- **Uncorrected age ±1σ:** the Cheng (2013) age-equation calculation with initial `(²³⁰Th/²³²Th)ₐ = 0`.
- **Bulk Earth age ±1σ:** the same equation using initial-Th `0.83 ± 0.43 (1σ)`, propagated with the measurement uncertainties.
- **IBIS age:** posterior median, the package's symmetric equivalent 1σ estimate, and equal-tailed 68% and 95% intervals.
- **Initial thorium:** IBIS posterior median, symmetric 1σ estimate, and 95% interval. The CSV also includes initial `(²³⁴U/²³⁸U)ₐ` and its uncertainty.
- The comparison plot displays analysis values by depth **without fitting an age–depth curve**. Conventional ages have horizontal bars of ±2σ; IBIS ages use the actual asymmetric equal-tailed 95% interval.

## Notes for deploying

The front end is static HTML/CSS/JS, but results require a running **Python** FastAPI backend with SciPy and IBIS. Do not publish the front end alone and expect calculations to work. For a multi-user public deployment, add authentication, durable job storage, worker process isolation, job expiry and rate limits before exposing the API.

## Quick check

```bash
python -m unittest discover -s tests -v
```

This executes the real Thoth prior and a deliberately tiny MCMC smoke test; it is not a scientific validation of the sampler.
