
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def median_with_uncertainty(values, errors, n_mc=100_000, seed=42):

    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    # remove NaNs
    good = np.isfinite(values) & np.isfinite(errors)
    values = values[good]
    errors = errors[good]

    rng = np.random.default_rng(seed)

    # ---------------------------------
    # Monte Carlo each measurement
    # ---------------------------------
    samples = rng.normal(
        loc=values,
        scale=errors,
        size=(n_mc, len(values))
    )

    # median for every Monte Carlo realization
    medians = np.median(samples, axis=1)

    # ---------------------------------
    # Median estimate
    # ---------------------------------
    median = np.median(medians)

    low, high = np.percentile(medians, [16, 84])

    # approximate symmetric 1-sigma
    median_err = (high - low) / 2

    return median, median_err, low, high
    
class IBIS_Data_Check:
    """
    The IBIS model assumes that data represent a closed system.
    IBIS is also jointly solving for the sysmtem. This means that any ages which are compromised could cause some problems for the inference.
    Here we try to build a simple check function that will read in the data and the bounds that are part of the original ibis model and then flag any data row that looks to be some other phenomena other than an initial thorium problem.
    
    
    Question
    --------
    What isnt an initial thorium issue?
        Out of order ages are those which do not follow the inferred stratigraphic relationship:
        Samples lower in depth are older for a Stalagmite.
        - Out of order ages, are assumed in IBIS to be the result of variable initial thorium that varies both spatially and temporally throughout the sample.
        - But out of order ages could occur because of other phenomena. So...
            - Analytical issues
            - Uranium mobility (loss and remobilization; Open system stuff)
            - Unknown stuff or other open system behaviour as well.
    

    """
    
    def __init__(self, data,ages, uncertainties):
        
        self.data = data
        self.ages = ages
        self.uncert = uncertainties
        self.r48_data = self.data["U234_U238_ratios"].to_numpy(float)
        self.r48_data_err = self.data["U234_U238_ratios_err"].to_numpy(float)
        self.U234_lam = 2.82203e-06
        
        
        self.r48 = (1.0 + (self.r48_data- 1.0) * np.exp(self.U234_lam  * self.ages))
        
        errs_r48 = np.zeros(self.r48_data.size)
        for i in range(len(errs_r48)):
            jac = np.array([1, self.U234_lam * (1.0 + (self.r48_data[i]- 1.0) * np.exp(self.U234_lam  * self.ages[i]))])
            cov = np.zeros((2,2))
            cov[0,0] = self.r48_data_err[i] ** 2
            cov[1,1] = self.uncert[i] ** 2

            err2 = jac @ cov @ jac.T
            errs_r48[i] = np.sqrt(err2)
        self.r48_err = errs_r48
        
    def u234_u238_check(self):
        """
        The best indication of beyond initial thorium behaviour
        is maybe the 234U/238U ratio
        """

        median, median_err, med_low, med_high = median_with_uncertainty(self.r48,self.r48_err)
        
        combined_sigma = np.hypot(self.r48_err, median_err)

        z = np.abs(self.r48 - median) / combined_sigma
        flag_uncertainty = z > 2
        
        fraction_difference = np.abs(self.r48 - median) / np.abs(median)
        percent_difference = fraction_difference * 100

        flag_10pct = fraction_difference > 0.10

        # require BOTH
        flag = flag_uncertainty & flag_10pct
        # Boolean array returned by the function
        flags = flag

        # --------------------------------------------------
        # Print index positions that are flagged
        # --------------------------------------------------
        flag_positions = np.where(flags)[0]

        # --------------------------------------------------
        # Print the corresponding DataFrame rows
        # --------------------------------------------------
        flagged_rows = self.data.iloc[flag_positions]

        print("\nFlagged rows:")
        print(flagged_rows)
        print("\nThese rows show beyong expected variance in 234U/238U",
              "\nI would double check this data and consider removing",
              "\nThen re-running IBIS")
        
        
        # ---------------------------------
        # Plot all results
        # ---------------------------------

        x = np.arange(len(self.r48))
        percent_threshold = 10
        # ±10% tolerance around median
        tolerance = (
            percent_threshold / 100
        ) * abs(median)

        tolerance_low = median - tolerance
        tolerance_high = median + tolerance


        fig, ax = plt.subplots(figsize=(10, 5))


        # ---------------------------------
        # ±10% tolerance band
        # ---------------------------------

        ax.axhspan(
            tolerance_low,
            tolerance_high,
            alpha=0.20,
            label=f"±{percent_threshold}% of median",
        )


        # ---------------------------------
        # Median uncertainty band
        # ---------------------------------

        ax.axhspan(
            med_low,
            med_high,
            alpha=0.35,
            label="Median 68% uncertainty",
        )


        # ---------------------------------
        # Median
        # ---------------------------------

        ax.axhline(
            median,
            linewidth=2,
            label=(
                f"Median = {median:.4f} "
                f"± {median_err:.4f}"
            ),
        )


        # ---------------------------------
        # Non-flagged measurements
        # ---------------------------------
        good = np.isfinite(self.r48) & np.isfinite(self.r48_err)

        not_flagged = good & ~flag

        ax.errorbar(
            x[not_flagged],
            self.r48[not_flagged],
            yerr=self.r48_err[not_flagged],
            fmt="o",
            capsize=3,
            label="Accepted",
        )


        # ---------------------------------
        # Flagged measurements
        # ---------------------------------

        flagged = good & flag

        if np.any(flagged):

            ax.errorbar(
                x[flagged],
                self.r48[flagged],
                yerr=self.r48_err[flagged],
                fmt="o",
                markersize=8,
                capsize=4,
                label="Flagged",
            )


        # ---------------------------------
        # Formatting
        # ---------------------------------

        ax.set_xlabel("Data point")
        ax.set_ylabel("Value")

        ax.set_xticks(x)

        ax.legend()

        plt.tight_layout()
        plt.show()
