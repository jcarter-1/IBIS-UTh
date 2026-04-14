from . import IBIS_Configuration
from . import IBIS_Bounds_and_Uncertainties
from . import IBIS_Thoth_V2
from . import IBIS_MCMC_Initial_Th_opt_test
from . import IBIS_stratv2
from . import USeries_Age_Equations
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, lognorm
from scipy.interpolate import interp1d

from datetime import datetime
from pathlib import Path


class IBIS:
    """
    Main class for running IBIS
    ============================
    Everyting should be acessible in here.
    
    Running Procedure
    -----------------
    Make an instance of this class
    e.g.,
    ibis_main_instance = IBIS_Main.IBIS(filepath to data,
    name for your sample, MCMC_samples = 5000, MCMC_burn_in = 1000, MCMC_Strat_samples = 50000,
                n_chains = 3, Start_from_pickles = True, Top_Age_Stal = False, Hiatus = False, show_bird = True,
                method = 'thoth',
                strat_resolution = 100)
    # IBIS inputs
    -------
    The input data is the filepath to the data. Data can be in csv or xlsx format. Columns should read:
    * U234_r
    * U234_r_err
    * Th230_r
    * Th230_r_err
    * Th232_r
    * Th232_r_err
    * Depth

    The three activity ratios and the sampling depth is needed. <br>
    Data should be at the 1sigma level. <br>
    Depth should be in the depth from top order.
    Depth uncertainty can be included or the model will add conservative uncertainty (Kinsley et al. 2025)
    
    All other inputs can be toggled.
    * MCMC_samples = Number of saved samples
    * MCMC_burn_in = Number of discarded samples from the initialization of the model
    * n_chains = Number of chains run (can be any integer value but 3 to 6 is good)
    * MCMC_Strat_samples = Number of stratigraphic age model samples (number is singular - burn in is half this number and is add to the total number - then these samples are discarded)
    * Start_from_pickles = True or False - Flag if you have run the sample before it will find the previous location and start from it
    * method = 'thoth' Choice of how to construct the prior - recommendation is 'thoth'

    ## Running Schematic

    * Begin with initialization of the model - e.g., intialize the classe object - called test below
    * Can look at some information
    
        * test.Get_IBIS_Bounds() - What are the ages and uncertainties when there is no initial thorium - requirement for maximum age bound.

        * test.df_reduced - Have a look at the dataframe - check numbers and ordering if necessary. If wrong reload data to meet above criteria

        * test.Set_Up_MCMC() - Get all neccessities for running MCMC

        * test.Plot_Priors() - Have a look at initial thorium prior for the speleothem

        * test.Run_MCMC() - Run the MCMC (Main part of the code)

        * test.Posterior_plot() - Reccomendation is to look at this - looking for signs of convergences - sampling around a mean, all chains look similar

        * test.Save_All() - If Chains look good data can be saved to the users computer - can then be retrieved for plotting elsewhere

        * test.Run_MCMC_Strat() - Second part if wanted - Get age and depth model relationship using IBIS-derived ages and uncertainties
    """
    
    def __init__(self, filepath, sample_name, MCMC_samples = 5000, MCMC_burn_in = 1000, MCMC_Strat_samples = 50000,
                n_chains = 3, Start_from_pickles = True, Top_Age_Stal = False,  show_bird = True,
                method = 'thoth',
                strat_resolution = 100,
                 diction_meta = None,
                 save_dir = None,
                 data_uncertainty_level = None):
                
        self.method = method
        self.filepath = filepath
        self.strat_resolution = strat_resolution
        # Configure the Data and uncertainty level
        self.data_uncertainty_level = data_uncertainty_level
        self.config = IBIS_Configuration.IBIS_Configuration_Class(self.filepath,
        data_uncertainty_level = self.data_uncertainty_level)
        # Get refined dataset
        self.df_reduced = self.config.Get_Measured_Ratios()
        # Sample
        self.sample_name = sample_name
        # Meta Data
        self.meta = dict(diction_meta) if diction_meta is not None else {}
        # Kdes_filepath
        self.kdes_name = self.sample_name + '_prior'
        # Bounds_Filename
        self.bounds_file_name = self.sample_name + '_bounds'
        # Number of samples
        self.MCMC_samples = MCMC_samples
        self.MCMC_burn_in = MCMC_burn_in
        # Number of Strat samples
        self.MCMC_Strat_samples = MCMC_Strat_samples
        # Number of chains
        self.n_chains = n_chains
        # Checks - boolean to be set in the model
        self.Include_Detrital = None
        self.are_there_bounds = False
        self.Thorium_prior_exist = False
        self.are_there_speleothem_parameters = False
        self.set_up_the_chain = False
        self.Chain_run = False
        self.Chain_strat_run = False
        # Set up save directory
        if save_dir is None:
            # Default: make folder in current working directory
            self.save_dir = Path.home() / "Desktop" / f"{sample_name}_folder"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(parents = True, exist_ok = True)
        print(f"All results will save to: {self.save_dir}")
        
        # Initialize
        self.Start_from_pickles = Start_from_pickles
        # Create a results folder
        self.rng = np.random.default_rng(1234)

        # Check for Hiatus and Top Stal Collection Age
        self.Top_Age_Stal = Top_Age_Stal
        if self.Top_Age_Stal: 
            self.collect_data = input('Please input date of sample collection in dd-mm-yyyy format: ')
            self.Top_Age_Stal  = self.Get_Top_Age()
        
        self.IBIS_intro = """
                 ==============    ==========     =============   ============
                        =          =         =          =         =
                        =          =          =         =         =
                        =          =          =         =         =
                        =          =         =          =         =
                        =          ==========           =         ============
                        =          =         =          =                    =
                        =          =          =         =                    =
                        =          =           =         =                    =
                        =          =         =          =                    =
                  =============    ==========      ============   ============

                Integrated Bayesian model for joint estimation of Initial thorium 
                correction and age-depth model for Speleothems
                
                """

        self.IBIS_BIRD = """             
                                                          ===
                                                        ========
                                                       ===========
                                                      =========  =======
                                                        ==            =========
                                                         ==                 =======
                                                          =                       ====
                                                           ==                         === 
                                                            ===                         ==
                                                             ===                         =
                                                            ====
                                            ====    ===   ===== 
                                       =====    ===       =====                         
                                   ====                      =
                               ====                        ======
                            =                               === 
                      =======                           ====
                   ====                              ===
                =====                              ===
               =                ========== =  =
                ===  ============         =     =   
                 ===========           =        =
                                    =           =
                                  =             =
                                 =              =
                                =               =
                               =                =
                               =                =
                              =                 =
                             =                    =
                            =                        =
                        =====                           =====
                    ===============                   ================
                    
            """
        if show_bird is True:
            print(self.IBIS_intro) 
            print(self.IBIS_BIRD)

    def Get_Top_Age(self): 
        if not self.Top_Age_Stal: 
            raise("Not applicable")
        if not isinstance(self.collect_data, str):
            raise TypeError("collect_data should be a string")  # Ensure it's a string
        try:
            collection_date = datetime.strptime(self.collect_data, "%d-%m-%Y")
            today = datetime.now()
            age = today.year - collection_date.year - ((today.month, today.day) < (collection_date.month, collection_date.day))
            return age
        except ValueError as e:
            raise ValueError(f"Date format error: {e}")  # Provide feedback on what went wrong


    def Get_IBIS_Input_Data(self): 
        Ibis_input = self.df_reduced

        return Ibis_input

    def Setup_Bounds_and_Uncertainties(self):
        
        bounds_path = self.save_dir / f"{self.bounds_file_name}.pkl"
        
        if bounds_path.exists() and self.Start_from_pickles == True:
            # If the file exists, load the bounds and uncertainties
            with open(bounds_path, 'rb') as input:
                self.bounds_params = pickle.load(input)
                print('Bounds and uncertainties file exists and is loaded.')
        else:
            # Initialize bounds and uncertainties
            # These bounds 0
            r08 = self.df_reduced['Th230_238U_ratios'].values
            r08_err = self.df_reduced['Th230_238U_ratios_err'].values
            r28 = self.df_reduced['Th232_238U_ratios'].values
            r28_err = self.df_reduced['Th232_238U_ratios_err'].values
            r48 = self.df_reduced['U234_U238_ratios'].values
            r48_err = self.df_reduced['U234_U238_ratios_err'].values
            
            self.bounds_ibis_ext = IBIS_Bounds_and_Uncertainties.IBIS_bounds_and_Uncertainties(r08, r28, r48,  r08_err, r28_err,
             r48_err, self.bounds_file_name,
            self.save_dir)

            
            self.bounds_ibis_ext.save_bounds()  # Ensure this method computes and saves the data to the file
            with open(bounds_path, 'rb') as f:
                self.bounds_params = pickle.load(f)
            print('Bounds and uncertainties computed and saved.')

        # Extract necessary bounds and uncertainties for further analysis
        self.test_ages = self.bounds_params[0]
        self.age_max = self.bounds_params[2]
        self.age_uncertainties = self.bounds_params[1]
        self.are_there_bounds = True
        
    def Get_IBIS_Bounds(self): 
        if not self.are_there_bounds: 
            self.Setup_Bounds_and_Uncertainties()
        
        return self.test_ages, self.age_max, self.age_uncertainties
        
    # ================================================
    # Thorium Prior
    # ================================================
    def _reset_thorium_cache(self):
        self._thor_inv_cdf = None
        self._thor_max = None

    def Initialize_Thoth(self):
        if not self.are_there_bounds:
            self.Setup_Bounds_and_Uncertainties()
        
        prior_fpath = self.save_dir / f"{self.kdes_name}.pkl"

        if prior_fpath.exists():
            with open(prior_fpath, 'rb') as f:
                self.Thor_prior = pickle.load(f)   # gaussian_kde
            print(f"Loaded existing Thorium prior from\n{prior_fpath}")
        else:
            self.thoth = IBIS_Thoth_V2.IBIS_Thoth_Robust(
                self.df_reduced, self.age_max,
               file_name=self.kdes_name, diction_meta = self.meta,
               save_dir = self.save_dir
            )
            self.thoth.save_thor_prior()
            with open(prior_fpath, 'rb') as f:
                self.Thor_prior = pickle.load(f)
            print(f"Computed & saved Thorium prior to\n {prior_fpath}")

        self.thor_kde = self.Thor_prior            # gaussian_kde
        self.Thorium_prior_exist = True            # <- consistent flag
        self._reset_thorium_cache()

    def Set_Up_MCMC(self):
        """
        Need to set up the Markov Chain Monte Carlo
        -------------------------------------------
        """
        if not getattr(self, "are_there_bounds", False):
            self.Setup_Bounds_and_Uncertainties()

        # One canonical flag name
        if not getattr(self, "Thorium_prior_exist", False):
            if self.method == 'thoth':
                self.Initialize_Thoth()
                
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.Thorium_prior_exist = True  # <- set this exact flag

        return self.thor_kde

    def _thor_pdf(self, x):
        if self.thor_kde is None:
            self.Set_Up_MCMC()

        x = np.asarray(x, dtype=float)
        x_eval = np.clip(x, 0.0, None)

        if hasattr(self.thor_kde, "pdf"):
            pdf = np.asarray(self.thor_kde.pdf(x_eval), dtype=float)
        elif callable(self.thor_kde):
            pdf = np.asarray(self.thor_kde(x_eval), dtype=float)
        else:
            raise TypeError("Thorium prior must be a scipy frozen rv, gaussian_kde, or callable pdf(x).")

        pdf[~np.isfinite(pdf)] = 0.0
        pdf[pdf < 0] = 0.0
        pdf[x < 0] = 0.0
        return pdf

    def _estimate_thor_xmax(self, x_min=0.0, q=0.9995, n_try=20000, max_rounds=8):
        """
        Estimate a sensible upper support bound for plotting / inverse-CDF sampling.
        """
        if self.thor_kde is None:
            self.Set_Up_MCMC()
            
        x_hi = 1.0
        for _ in range(max_rounds):
            grid = np.linspace(x_min, x_hi, 1024)
            pdf = self._thor_pdf(grid)
            peak = np.nanmax(pdf)
            tail = pdf[-1]
            if np.isfinite(peak) and peak > 0 and tail / peak < 1e-6:
                return float(x_hi)
            x_hi *= 2.0

        return float(x_hi)

    def _build_thor_inv_cdf(self, x_min=0.0, x_max=None, grid_points=4096):
        """
        Build inverse CDF on a finite support for robust sampling from any pdf-like prior.
        """
        if self.thor_kde is None:
            self.Set_Up_MCMC()

        if x_max is None:
            x_max = self._estimate_thor_xmax(x_min=x_min)

        x_grid = np.linspace(float(x_min), float(x_max), int(grid_points))
        pdf = self._thor_pdf(x_grid)

        area = np.trapz(pdf, x_grid)
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Could not build thorium inverse CDF: PDF integrates to zero or invalid value.")

        pdf /= area

        dx = np.diff(x_grid)
        cdf = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
        ))
        cdf /= cdf[-1]

        keep = np.concatenate(([True], np.diff(cdf) > 0))
        cdf = cdf[keep]
        x_grid = x_grid[keep]

        self._thor_inv_cdf = interp1d(
            cdf,
            x_grid,
            bounds_error=False,
            fill_value=(float(x_min), float(x_max))
        )
        self._thor_xmax = float(x_max)

    def _thor_rvs(self, n):
        if self.thor_kde is None:
            self.Set_Up_MCMC()

        n = int(n)
        if n <= 0:
            return np.array([], dtype=float)

        # frozen scipy distribution
        if hasattr(self.thor_kde, "rvs"):
            try:
                s = np.asarray(self.thor_kde.rvs(size=n, random_state=self.rng), dtype=float)
            except TypeError:
                s = np.asarray(self.thor_kde.rvs(size=n), dtype=float)

            s = s[np.isfinite(s)]
            s = s[s >= 0]
            return s

        # gaussian_kde native sampler
        if hasattr(self.thor_kde, "resample"):
            out = []
            need = n

            while need > 0:
                draw = np.asarray(self.thor_kde.resample(max(need * 2, 1000)), dtype=float).reshape(-1)
                draw = draw[np.isfinite(draw)]
                draw = draw[draw >= 0]
                if draw.size:
                    take = draw[:need]
                    out.append(take)
                    need -= take.size

            return np.concatenate(out)

        # generic callable pdf -> inverse CDF sampler
        if self._thor_inv_cdf is None:
            self._build_thor_inv_cdf()

        u = self.rng.random(n)
        s = np.asarray(self._thor_inv_cdf(u), dtype=float)
        s = s[np.isfinite(s)]
        s = s[s >= 0]
        return s

    def Generate_samples_from_Prior(self, n=100_000):
        return self._thor_rvs(n)

    def Plot_Priors(self, n_plot=50000, q_low=0.001, q_high=0.999, grid_points=1000):
        if self.thor_kde is None:
            self.Set_Up_MCMC()

        # choose plotting range robustly
        if hasattr(self.thor_kde, "ppf"):
            try:
                lo = max(0.0, float(self.thor_kde.ppf(q_low)))
                hi = float(self.thor_kde.ppf(q_high))
            except Exception:
                s = self._thor_rvs(n_plot)
                lo = max(0.0, float(np.quantile(s, q_low)))
                hi = float(np.quantile(s, q_high))
        else:
            s = self._thor_rvs(n_plot)
            lo = max(0.0, float(np.quantile(s, q_low)))
            hi = float(np.quantile(s, q_high))

        if not np.isfinite(hi) or hi <= lo:
            hi = self._estimate_thor_xmax(x_min=0.0)

        x = np.linspace(lo, hi, grid_points)
        y = self._thor_pdf(x)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))
        ax.plot(
            x, y, lw=1.8,
            label=r"Prior for $^{230}\mathrm{Th}/^{232}\mathrm{Th}_{\mathrm{init}}$",
            color='dodgerblue'
        )
        ax.fill_between(x, y, alpha=0.35, color='navy')
        ax.set_xlabel(r"$^{230}$Th/$^{232}$Th initial")
        ax.set_ylabel("Density")
        ax.set_xlim(lo, hi)
        ax.legend(frameon=True)
        return fig, ax
        
        
    def Look_at_initial(self):
        if not self.set_up_the_chain:
            self.thor_kde = self.Set_Up_MCMC()

        self.Ibis_Chains = IBIS_MCMC_Initial_Th_opt_test.IBIS_MCMC(self.thor_kde,
                                              self.age_max,
                                              self.age_uncertainties,
                                              self.df_reduced,
                                              iterations = self.MCMC_samples,
                                              burn_in = self.MCMC_burn_in,
                                              sample_name = self.sample_name,
                                              n_chains = self.n_chains,
                                              Start_from_pickles = self.Start_from_pickles,
                                              method = self.method,
                                              save_dir = self.save_dir)
                                              
        return self.Ibis_Chains.Initial_Guesses_for_Model()
                
    def Run_MCMC(self): 
        if not self.set_up_the_chain: 
            self.thor_kde = self.Set_Up_MCMC()

        self.Ibis_Chains = IBIS_MCMC_Initial_Th_opt_test.IBIS_MCMC(self.thor_kde,
                                              self.age_max,
                                              self.age_uncertainties,
                                              self.df_reduced,
                                              iterations = self.MCMC_samples,
                                              burn_in = self.MCMC_burn_in,
                                              sample_name = self.sample_name,
                                              n_chains = self.n_chains,
                                              Start_from_pickles = self.Start_from_pickles,
                                              method = self.method,
                                              save_dir = self.save_dir)

        self.Ibis_Chains.Run_MCMC();
        self.Chain_run = True

    def Get_MCMC_Results(self): 

        self.Run_MCMC()
        self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()

    def Get_Post_Vals(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()
        self.Ibis_Chains.Get_Posterior_Values()
        
    def Posterior_plot(self):         
        
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()
        self.Ibis_Chains.Get_Posterior_plot()

    def Model_U_ages(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()

        age_c, (age_lo68, age_hi68), (age_lo95, age_hi95), age_sigma = self.Ibis_Chains.Get_Useries_Ages()

        return age_c, (age_lo68, age_hi68), (age_lo95, age_hi95), age_sigma

    def Model_Initial_Thorium(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()


        Th0_c, (Th0_lo68, Th0_hi68), (Th0_lo95, Th0_hi95), Th0_sigma = self.Ibis_Chains.Get_Initial_Thoriums()

        return Th0_c, (Th0_lo68, Th0_hi68), (Th0_lo95, Th0_hi95), Th0_sigma

    def Model_Initial_U234(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()

        U0_c,  (U0_lo68,  U0_hi68),  (U0_lo95,  U0_hi95),  U0_sigma = self.Ibis_Chains.Get_234U_initial()

        return U0_c,  (U0_lo68,  U0_hi68),  (U0_lo95,  U0_hi95),  U0_sigma

    def Plot_U_ages_and_Age_model(self, AGE_MODEL = True): 
        self.model_ages, self.model_ages_err, self.model_depths, self.model_depths_err = self.Model_Ages_Depths()
        self.U_series_ages, self.U_series_ages_err  = self.Model_U_ages()

        
        fig, ax = plt.subplots(1,1, figsize = (5,7))
        ax.errorbar(x = self.U_series_ages,
            xerr = self.U_series_ages_err[0], 
                y = self.df_reduced['Depths'].values,
                fmt = 'o',
               label = 'U Series\nModel', 
               alpha = 1, 
                markerfacecolor = 'dodgerblue',
            ecolor = 'k',
           markersize = 10, 
           markeredgecolor = 'k')

        if AGE_MODEL:
        
            ax.fill_betweenx(self.model_depths, self.model_ages - self.model_ages_err[0], 
                       self.model_ages_err[1] + self.model_ages,
                       label = 'Age-Depth\nModel', 
                       alpha = 0.5)
        
            ax.set_ylabel('Depth (mm)')
            
        
        ax.set_xlabel('Apparent Age (a)')
        ax.set_ylim(self.df_reduced['Depths'].values.max() + 5, 
                    self.df_reduced['Depths'].values.min() - 5)
        
        ax.legend(loc = 3)
        
    def SaveSummary(self):
        self.Ibis_Chains.SummaryDataFrame()
        print(f'Summary saved to {self.sample_name}_ibis_summary.csv')
        
    def MakeCompleteDataFrame(self):
        """
        This is going to be the the full data frame
        * Measured activity ratios and uncertainties
        * Uncorrected ages and uncertainties
        * Depths measured and uncertainties
        * IBIS outputs
        """
        df_ibis = self.Ibis_Chains.MakeSummaryDataFrame()
        df_data = self.df_reduced
        df_combined = df_ibis.join(df_data)
        uncorr_ages, uncorr_ages_err, _ = self.Get_IBIS_Bounds()
        df_combined['Uncorrected age (a)'] = uncorr_ages
        df_combined['Uncorrected age 1sigma (a)'] = uncorr_ages_err
        
        return df_combined
        
    def SaveTotalDataFrame(self):
        df_all = self.MakeCompleteDataFrame()
        output_path = self.save_dir /f"{self.sample_name}_ibis_complete_summary.csv"
        df_all.to_csv(output_path, index=False)
        print(f"Complete summary saved to: {output_path}")

    # These don't exist - but I will add them
    # Existed in earlier version that needs some updating
    
    #def Get_Chain_Stats_Thor(self):
    #    return self.Ibis_Chains.In_Thor_Chain_Stats()

    #def Get_In_Thor(self):
    #    return self.Ibis_Chains.Get_Initial_Thoriums()
        
    #def Get_Chain_Stats_Uages(self):
    #    return self.Ibis_Chains.Useries_Age_Chain_Stats()

    #def Get_Chain_Stats_Lam_U234(self):
    #    return self.Ibis_Chains.lam234_Chain_Stats()

    #def Get_Chain_Stats_Lam_Th230(self):
    #    return self.Ibis_Chains.Th230_Chain_Stats()
    
    def preprocess_boundary_ages(self,
        median,
        low68,
        high68,
        low95,
        high95,
        age_floor=0.0,
        tol=1e-10,
        use_divisor=2.0,
    ):
        """
        Convert summarized posterior intervals into asymmetric age errors suitable
        for the age-depth model.

        Returns
        -------
        age_med : ndarray
        age_err_low : ndarray
        age_err_high : ndarray
        flags : ndarray of bool
            True where a boundary-collapsed 68% interval was replaced using 95%.
        """
        median = np.asarray(median, float)
        low68 = np.asarray(low68, float)
        high68 = np.asarray(high68, float)
        low95 = np.asarray(low95, float)
        high95 = np.asarray(high95, float)

        age_med = median.copy()
        age_err_low = np.maximum(age_med - low68, 0.0)
        age_err_high = np.maximum(high68 - age_med, 0.0)

        flags = np.zeros(age_med.shape, dtype=bool)

        for i in range(age_med.size):
            collapsed68 = (
                abs(age_med[i] - age_floor) < tol and
                abs(low68[i]   - age_floor) < tol and
                abs(high68[i]  - age_floor) < tol
            )

            if collapsed68 and (high95[i] > age_floor + tol):
                age_err_low[i] = 0.0
                age_err_high[i] = max((high95[i] - age_floor) / use_divisor, tol)
                flags[i] = True

        return age_med, age_err_low, age_err_high, flags

    def Load_U_Series_Ages(self, filename): 
        df = pd.read_csv(filename)
        U_series_ages = df['age'].values
        U_series_ages_err_low68 = df['age_lo68'].values
        U_series_ages_err_high68 = df['age_hi68'].values
        U_series_ages_err_low95 = df['age_lo95'].values
        U_series_ages_err_high95 = df['age_hi95'].values
        # Return ages and a tuple of their error bounds (low, high)
        U_age_median, U_age_err_low, U_age_err_high, _ = self.preprocess_boundary_ages(U_series_ages,
                                                        U_series_ages_err_low68,
                                                        U_series_ages_err_high68,
                                                        U_series_ages_err_low95,
                                                        U_series_ages_err_high95)
                                                        
        return U_age_median, (U_age_err_low, U_age_err_high)

    def Run_MCMC_Strat(self): 
        u_ages_file =  self.save_dir / f'{self.sample_name}_ibis_summary.csv'
        import pandas as pd

        if os.path.exists(u_ages_file): 
            print(f"File '{u_ages_file}' exists. Skipping initial MCMC and running stratigraphy MCMC directly. Sit Tight. Time for a cup of tea.")
            self.U_series_ages, self.U_series_ages_err = self.Load_U_Series_Ages(u_ages_file)
        else: 
            print("Bayesian Ages Dont Exist Yet! Running IBIS Part1. Hold on...")
            self.Run_MCMC()
            self.SaveSummary()
            u_ages_file =  self.save_dir / f'{self.sample_name}_ibis_summary.csv'
            if os.path.exists(u_ages_file):
                print(f"File '{u_ages_file}' exists. Loading")
            self.U_series_ages, self.U_series_ages_err = self.Load_U_Series_Ages(u_ages_file)
    
        U_ages = self.U_series_ages
        U_ages_low = self.U_series_ages_err[0]
        U_ages_high = self.U_series_ages_err[1]
        
        self.Ibis_Stratigraphy = IBIS_stratv2.IBIS_Strat2(U_ages,
                                               U_ages_low,
                                               U_ages_high,
                                               self.df_reduced, 
                                                self.sample_name,
                                               self.Start_from_pickles, 
                                               self.n_chains, 
                                               iterations = self.MCMC_Strat_samples, 
                                               burn_in = int(self.MCMC_Strat_samples)/2, 
                                                       resolution=100,
                                                top_age=self.meta.get("top_age", None),
                                                top_age_err=self.meta.get("top_age_err", None),
                                                top_age_depth=self.meta.get("top_age_depth", None),
                                                model_top_depth=self.meta.get("model_top_depth", None),
                                                model_bottom_depth=self.meta.get("model_bottom_depth", None),
                                                pad_frac=0.10,
                                                sigma_extra=0.0,
                                                smoothness=self.meta.get("smoothness", 1e-6),
                                                save_dir=self.save_dir)
        
        self.Ibis_Stratigraphy.Run_MCMC_Strat();
        self.Chain_strat_run = True

    def Age_Model(self): 
        if not self.Chain_strat_run: 
            self.Run_MCMC_Strat()
            self.results_dicts  = self.Ibis_Stratigraphy.Get_Results_Dictionary()

        self.age_low, self.age_high, self.age_median = self.Ibis_Stratigraphy.Get_Age_Model()

        return self.age_low, self.age_high, self.age_median

    def Depth_Model(self): 
        if not self.Chain_strat_run: 
            self.Run_MCMC_Strat()
            self.results_dicts  = self.Ibis_Stratigraphy.Get_Results_Dictionary()


        self.depth_low, self.depth_high, self.depth_median = self.Ibis_Stratigraphy.Get_Depth_Model()

        return self.depth_low, self.depth_high, self.depth_median

    def Get_Age_Depth_Plot(self): 
        self.Ibis_Stratigraphy.Get_Age_Depth_Plot()

    def Save_Age_Depth_Model(self): 
        self.Ibis_Stratigraphy.Save_Age_Depth_Model()
        print(f'Age/Depth model saved to: {self.sample_name}_Age_Depth_Model.xlsx')
        
