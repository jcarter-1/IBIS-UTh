from . import USeries_Age_Equations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.interpolate import interp1d
import matplotlib as mpl
import os
import pickle
from pathlib import Path


class U_Series_Age_Equation:
    def __init__(self, r08, r08_err,
                 r28, r28_err,
                 r48, r48_err):
        
        self.r08 = r08
        self.r08_err = r08_err
        self.r28 = r28
        self.r28_err = r28_err
        self.r48 = r48
        self.r48_err = r48_err
        self.lambda_230 = 9.1577e-6
        self.lambda_234 = 2.8263e-6
        self.lambda_230_err = 1.3914e-8
        self.lambda_234_err = 2.8234e-9
        self.r02_initial = 0.0
        self.r02_initial_err = 0.0

    

    def Age_Equation(self, T):
    
        A = self.r08 - self.r28 * self.r02_initial * np.exp(-self.lambda_230 * T)
        B = 1 - np.exp(-self.lambda_230 * T)
        D = self.r48 - 1
        lam_diff =self.lambda_230 - self.lambda_234
        E = self.lambda_230 / lam_diff
        F = 1 - np.exp(-lam_diff*T)
        C = D * E * F
        return A - B - C


    def Age_solver(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]


    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        """
        Age and uncertainty calculation
        decay constant uncertainties not included here
        """
        Age = self.Age_solver()
        
        # Compute lambda difference
        lam_diff = self.lambda_234 - self.lambda_230
        
        # Compute df/dT components
        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = - (self.r48 - 1) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3
        
        # Compute partial derivatives dt/dx_i
        dt_dr08 = -1 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = - ((self.lambda_230 / lam_diff) * (1 - np.exp(lam_diff * Age))) / df_dT
    
        age_jacobian = np.array([dt_dr08,
                                 dt_dr28,
                                 dt_dr02,
                                 dt_dr48])
    
        cov_age = np.zeros((4,4))
        cov_age[0,0] = self.r08_err**2
        cov_age[1,1] = self.r28_err**2
        cov_age[2,2] = self.r02_initial_err**2
        cov_age[3,3] = self.r48_err**2
    
        age_err =  age_jacobian @ cov_age @ age_jacobian.T
    
        return Age, np.sqrt(age_err)


class IBIS_bounds_and_Uncertainties: 
    """
    This function produced neccessary bounds and uncertaintis for 
    the IBIS model 
    -------------------------------------------------------------

    Returns
    -------
    - Age uncertainties (w/ decay constant uncertainties)
    - Maximum age of the speleothem
    - Minimum and Maximum Thorium bounds
    """
    def __init__(self, r08, r28, r48, r08_err, r28_err, r48_err, bounds_filename, save_dir = None):
        self.r08 = r08
        self.r08_err = r08_err
        self.r28 = r28
        self.r28_err = r28_err
        self.r48 = r48
        self.r48_err = r48_err
        self.lam_230 = 9.17055e-06  # Cheng et al. (2013)
        self.lam_230_err =6.67e-09  # Cheng et al. (2013)
        self.lam_234 = 2.82203e-06  # Cheng et al. (2013)
        self.lam_234_err =  1.494e-09  # Cheng et al. (2013)
        self.max_age = None
        self.uncertainties = None
        self.Check_Bounds_and_uncertainties = False # Initially flag as fall untill all bounds and uncertainties are calculated
        self.ages = None
        self.Bounds_ = None
        self.bounds_filename = bounds_filename
        # NEW: store save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = Path.cwd()

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def Age_Calc_NoInitialTh(self):
        """
        Compute age ± uncertainty for each depth using
        U_Series_Age_Equation with r02_initial=0.
        """
        ages = []
        uncerts = []
        for i in range(len(self.r08)):
            eq = U_Series_Age_Equation(
                # core measurements
                r08               = self.r08[i],
                r08_err           = self.r08_err[i],
                r28               = self.r28[i],
                r28_err           = self.r28_err[i],
                r48               = self.r48[i],
                r48_err           = self.r48_err[i],
)
            age, err = eq.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
            ages.append(age)
            uncerts.append(err)

        self.ages      = np.array(ages)
        self.uncertainties = np.array(uncerts)
        return self.ages, self.uncertainties

    def Maximum_Age(self):
        if self.ages is not None:
            self.max_age = self.ages[-1] + 5*self.uncertainties[-1]
        return self.max_age

    def Get_Bounds(self): 
        # Make sure we have all the bounds
        self.ages, self.uncertainties= self.Age_Calc_NoInitialTh()
        self.max_age = self.Maximum_Age()
        self.Bounds_ = (self.ages, self.uncertainties, self.max_age)

    def save_bounds(self):

        if self.Bounds_ is None:
            self.Get_Bounds()

        full_path = self.save_dir / f"{self.bounds_filename}.pkl"

        with open(full_path, 'wb') as f:
            pickle.dump(self.Bounds_, f)

        print(f"Ages, uncertainties, and maximum age saved to {full_path}")

        return self.Bounds_

    def load_bounds(self):

        full_path = self.save_dir / f"{self.bounds_filename}.pkl"

        if full_path.exists():
            with open(full_path, 'rb') as f:
                self.Bounds_ = pickle.load(f)
            print("Bounds loaded from file")
        else:
            print("Bounds do not exist yet, generating...")
            self.save_bounds()
