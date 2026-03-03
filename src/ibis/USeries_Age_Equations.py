
import numpy as np 
from scipy.optimize import fsolve



class USeries_ages: 
    """
    Helper Class to solve age equation
    needs the corrected 230Th/238U and corrected 234U/238U ratio
    and the decay constant
    these will come from somewhere else and then this computes the age
    -------------------------------------------------------------------
    Ages are determined using fsolve a package in scipy.optimize
    """
    def __init__(self, r08, r48, lam_230, lam_234): 
        self.r08 = r08
        self.r48 = r48
        self.lam_230 = lam_230
        self.lam_234 = lam_234

    """ 
    Age Equation Stuff
    """
    def Age_Equation(self, age): 

        
        """
        Internal method to compute the left and right sides of the U-series age equation.
        """

        left_side = self.r08
        right_side_part1 = 1 - np.exp(-self.lam_230 * age)
        right_side1 = 1 - np.exp(-self.lam_230* age)
        right_side_part2 = (self.r48 - 1)
        right_side_part3 = (self.lam_230 / (self.lam_234 - self.lam_230))
        right_side_part4 = (1 - np.exp((self.lam_234 - self.lam_230) * age))
        right_side2 = right_side_part2 * right_side_part3 * right_side_part4
    
        return left_side - right_side1 + right_side2

    def Age_solver(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]


    def U_ages(self, age_guess = 1e4):
        """
        Calculate U-series ages for multiple sets of initial conditions using a vectorized approach.
        """
        U_series_ages = np.array([self.Age_solver(age_guess) for _ in range(len(self.r08))])
        return U_series_ages


    def Age_Equation_all(self, age, r08_i, r48_i): 
        
        left_side = r08_i
        right_side_part1 = 1 - np.exp(-self.lam_230 * age)
        right_side1 = 1 - np.exp(-self.lam_230* age)
        right_side_part2 = (r48_i - 1)
        right_side_part3 = (self.lam_230 / (self.lam_234 - self.lam_230))
        right_side_part4 = (1 - np.exp((self.lam_234 - self.lam_230) * age))
        right_side2 = right_side_part2 * right_side_part3 * right_side_part4
    
        return left_side - right_side1 + right_side2


    def Age_solver_all(self, r08_i, r48_i, age_guess=1e4):
        """
        Solves the age equation for a single pair of r08 and r48 values.
        """
        func = lambda age: self.Age_Equation_all(age, r08_i, r48_i)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]

    def U_ages_all(self, age_guess=1e4):
        """
        Calculates U-series ages for each pair of r08 and r48 values.
        """
        U_series_ages = np.array([
            self.Age_solver_all(r08_i, r48_i, age_guess)
            for r08_i, r48_i in zip(self.r08, self.r48)
        ])
        return U_series_ages



