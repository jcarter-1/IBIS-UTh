from . import IBIS_Configuration
from . import IBIS_Bounds_and_Uncertainties
from . import IBIS_Thoth_V2
from . import IBIS_MCMC_Initial_Th_opt_test
from . import IBIS_stratv2
from . import USeries_Age_Equations
from . import IBIS_strat_Hiatus
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, lognorm
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
    * method = 'thoth' or 'bulkspeleothem' - Choice of how to construct the prior - recommendation is 'thoth'

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
                n_chains = 3, Start_from_pickles = True, Top_Age_Stal = False, Hiatus = False, show_bird = True,
                method = 'thoth',
                strat_resolution = 100,
                 diction_meta = None,
                 save_dir = None):
                
        self.method = method
        self.filepath = filepath
        self.strat_resolution = strat_resolution
        # Configure the Data
        self.config = IBIS_Configuration.IBIS_Configuration_Class(self.filepath)
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
        self.Hiatus_Check = Hiatus

        if self.Hiatus_Check:
            while True:
                try:
                    self.Hiatus_start = float(input(
                        "Please input start (highest point) of the observed Hiatus "
                        "– match units of the input data: "
                    ))
                    break
                except ValueError:
                    print("⚠️  Invalid number. Please enter a decimal or integer value.")
        
            while True:
                try:
                    self.Hiatus_end = float(input(
                        "Please input end (lowest point) of the observed Hiatus "
                        "– match units of the input data: "
                    ))
                    break
                except ValueError:
                    print("⚠️  Invalid number. Please enter a decimal or integer value.")
        
        
        
        self.IBIS_intro = """
                 ==============    ==========     =============   ============
                        =          =         =          =         =
                        =          =          =         =         =
                        =          =          =         =         =
                        =          =         =          =         =
                        =          ==========           =         ============
                        =          =         =          =                    =
                        =          =          =         =                    =
                        =          =          =         =                    =
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
            
            self.bounds_ibis_ext = IBIS_Bounds_and_Uncertainties.IBIS_bounds_and_Uncertainties(r08, r28, r48,
            r08_err, r28_err, r48_err, self.bounds_file_name,
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

    def Initialize_Thoth(self):
        prior_fpath = self.save_dir / f"{self.kdes_name}.pkl"

        if prior_fpath.exists():
            with open(prior_fpath, 'rb') as f:
                self.Thor_prior = pickle.load(f)   # gaussian_kde
            print(f"♻️  Loaded existing Thorium prior from\n{prior_fpath}")
        else:
            self.thoth = IBIS_Thoth_V2.IBIS_Thoth_Robust(
                self.df_reduced, self.age_max,
               file_name=self.kdes_name, diction_meta = self.meta,
               save_dir = self.save_dir
            )
            self.thoth.save_thor_prior()
            with open(prior_fpath, 'rb') as f:
                self.Thor_prior = pickle.load(f)
            print(f"✅  Computed & saved Thorium prior to\n {prior_fpath}")

        self.thor_kde = self.Thor_prior            # gaussian_kde
        self.Thorium_prior_exist = True            # <- consistent flag


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
            elif self.method == 'bulkspeleothem':
                # Your parameters:
                # lognorm(s, loc, scale) with s = sigma_log, loc = 0, scale = exp(mu_log)
                # You currently use: lognorm(1.25748898, 0.0, 2.7024700608392247)
                self.thor_kde = lognorm(
                    1.2574889802170042,  # s (σ in log-space)
                    0.0,                 # loc
                    2.7024700608392247   # scale = exp(μ)
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.Thorium_prior_exist = True  # <- set this exact flag

        return self.thor_kde

        
    def _thor_pdf(self, x):
        x = np.asarray(x, float)
        if hasattr(self.thor_kde, "pdf"):         # e.g., scipy.stats.lognorm (frozen)
            return self.thor_kde.pdf(x)
        elif hasattr(self.thor_kde, "__call__"):  # gaussian_kde
            return np.asarray(self.thor_kde(x), float)
        else:
            raise TypeError("Thor_KDE must be a scipy frozen rv or gaussian_kde")
    
    def _thor_rvs(self, n):
        # 1) scipy rv: use its sampler
        if hasattr(self.thor_kde, "rvs"):
            return np.asarray(self.thor_kde.rvs(size=n), float)
        # 2) gaussian_kde: prefer native resample if available (fast & correct)
        if hasattr(self.thor_kde, "resample"):
            s = self.thor_kde.resample(n)  # shape (dim, n); for 1D dim=1
            return np.asarray(s).reshape(-1)
        # 3) fallback: inverse-CDF on a grid
        if getattr(self, "_thor_inv_cdf", None) is None:
            self._build_thor_inv_cdf()
        u = self.rng.random(n)
        return np.asarray(self._thor_inv_cdf(u), float)
    
    def _build_thor_inv_cdf(self, x_min=0.0, x_max=None, grid_points=4096):
        # only needed for KDE fallback
        if hasattr(self.thor_kde, "rvs"):  # parametric: skip
            return
        # adapt support until ~all mass is inside
        xmax = 0.5 if x_max is None else float(x_max)
        for _ in range(12):
            x_grid = np.linspace(x_min, xmax, grid_points)
            pdf = np.clip(self._thor_pdf(x_grid), 0.0, None)
            dx = x_grid[1] - x_grid[0]
            cdf = np.cumsum(pdf) * dx
            tot = cdf[-1]
            if tot <= 0 or not np.isfinite(tot):
                xmax *= 2.0
                continue
            cdf /= tot
            if cdf[-1] > 0.999:
                break
            xmax *= 2.0
        from scipy.interpolate import interp1d
        self._thor_inv_cdf = interp1d(cdf, x_grid, bounds_error=False,
                                    fill_value=(x_min, xmax))
                        
    def Generate_samples_from_Prior(self, n=100_000):
        return self._thor_rvs(n)

    def Plot_Priors(self, smooth_sigma_px=50):

    
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.8))
        ax.plot(self.thor_kde.x,
               self.thor_kde.y,
               lw=1.8, label=r"Prior for $^{230}\mathrm{Th}/^{232}\mathrm{Th}_{\mathrm{init}}$",
               color= 'dodgerblue')
        ax.fill_between(self.thor_kde.x,
               self.thor_kde.y, alpha=0.35, color = 'navy')
        #ax.set_xlim(lo, hi)                 # <-- use lo, hi (no hard 0)
        #ax.set_ylim(0.0, ymax + ypad)       # <-- give headroom
        ax.set_xlabel(r"$^{230}$Th/$^{232}$Th initial")
        ax.set_ylabel("Density")
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
        
    def Get_Chain_Stats_Thor(self):
        return self.Ibis_Chains.In_Thor_Chain_Stats()

    def Get_In_Thor(self): 
        return self.Ibis_Chains.Get_Initial_Thoriums()
        
    def Get_Chain_Stats_Uages(self): 
        return self.Ibis_Chains.Useries_Age_Chain_Stats()

    def Get_Chain_Stats_Lam_U234(self): 
        return self.Ibis_Chains.lam234_Chain_Stats()

    def Get_Chain_Stats_Lam_Th230(self): 
        return self.Ibis_Chains.Th230_Chain_Stats()


    def Load_U_Series_Ages(self, filename): 
        df = pd.read_csv(filename)
        U_series_ages = df['age'].values
        U_series_ages_err_low = df['age_err_lo68'].values
        U_series_ages_err_high = df['age_err_hi68'].values
        # Return ages and a tuple of their error bounds (low, high)
        return U_series_ages, (U_series_ages_err_low, U_series_ages_err_high)     

    def Run_MCMC_Strat(self): 
        u_ages_file =  self.save_dir / f'{self.sample_name}_ibis_summary.csv'
        import pandas as pd

        if os.path.exists(u_ages_file): 
            print(f"File '{u_ages_file}' exists. Skipping initial MCMC and running stratigraphy MCMC directly. Sit Tight. Time for a cup of tea.")
            self.U_series_ages, self.U_series_ages_err = self.Load_U_Series_Ages(u_ages_file)
        else: 
            print("Bayesian Ages Dont Exist Yet! Running IBIS Part1. Hold on...")
            self.U_series_ages, self.U_series_ages_err, _, _  = self.Model_U_ages()
    

        U_ages = self.U_series_ages
        U_ages_low = self.U_series_ages_err[0]
        U_ages_high = self.U_series_ages_err[1]
        
        self.Ibis_Stratigraphy = IBIS_stratv2.IBIS_Strat(U_ages,
                                               U_ages_low,
                                               U_ages_high,
                                               self.df_reduced, 
                                                self.sample_name,
                                               self.Start_from_pickles, 
                                               self.n_chains, 
                                               iterations = self.MCMC_Strat_samples, 
                                               burn_in = int(self.MCMC_Strat_samples)/2, 
                                               Top_Age_Stal = self.Top_Age_Stal,
                                               resolution = self.strat_resolution,
                                               save_dir = self.save_dir)
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
        
