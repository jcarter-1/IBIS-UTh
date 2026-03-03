
import numpy as np
import pandas as pd
import os
import pickle

class IBIS_Data_Format:
    """
    Read in data in anyformat and put it into IBIS format
    Read in the ratios in any format and we want to get back to the true activity ratios
    Clean data before hand
    Everyting should be a float
    Uncertainties at 1 sigma
    """
    def __init__(self, sample_name = 'SAMPLE', r08 = None, r08_err = None, r28 = None,  r28_err = None, r02 = None, r02_err = None, r48 = None, r48_err = None, d48 = None, d48_err = None, th232 = None, th232_err = None, u238 = None, u238_err = None,
        depths = None, measurement_name =None):
    
        # INITIALIZE
        self.sample = sample_name
        self.r08 = r08
        self.r08_err = r08_err
        
        self.r28 = r28
        self.r28_err = r28_err
        
        self.r02 = r02
        self.r02_err = r02_err
        
        self.r48 = r48
        self.r48_err = r48_err
        
        self.d48 = d48
        self.d48_err = d48_err
        
        
        # Sample concentration
        self.u238 = u238
        self.u238_err = u238_err
        self.th232 = th232
        self.th232_err = th232_err
        
        self.N_data = len(self.r08)
        self.depths = depths
        self.measurement_name = measurement_name
        self.lam238 = np.log(2)/4.4683e9
        self.lam238_err = (0.0024/4.4683) * self.lam238
        self.lam232 = np.log(2)/1.41e10
        self.lam232_err = (1/100) * self.lam232 # 1% as quoted from Farley et al.
        
    
    def Convert_to_r48(self):
        u234_u238_act_ratio = (self.d48/1000) + 1
        rel_err = self.d48_err/self.d48
        
        return u234_u238_act_ratio, u234_u238_act_ratio * rel_err
        
    def Convert_u_and_th_r28(self):
        th232_u238_ratio = self.th232/self.u238
        
        th230_u238_ratio_var = np.zeros(self.N_data)
        for i in range(self.N_data):
            jac = np.array([1/self.u238[i], -self.th232[i]/self.u238[i]**2])
            cov = np.zeros((2,2))
            cov[0,0] = self.th232_err[i]**2
            cov[1,1] = self.u238_err[i]**2
            
            th230_u238_ratio_var[i] = jac @ cov @ jac.T
            
        th232_u238_act_ratio = th232_u238_ratio * (self.lam232 /self.lam238)
        
        
        th232_u238_act_ratio_var = np.zeros(self.N_data)
        
        for i in range(self.N_data):
            jac = np.array([(self.lam232 /self.lam238), th232_u238_ratio[i]/self.lam238, - th232_u238_ratio[i]*self.lam232/self.lam238**2])
            
            cov = np.zeros((3,3))
            cov[0,0] = th230_u238_ratio_var[i]
            cov[1,1] = self.lam232_err**2
            cov[2,2] = self.lam238_err**2
            
            th232_u238_act_ratio_var[i] = jac @ cov @ jac.T
            
        return th232_u238_act_ratio, np.sqrt(th232_u238_act_ratio_var)
        
    def Convert_to_r28(self):
        th230_u238_act_ratio = self.r08 / self.r02
        
        th230_u238_act_ratio_var = np.zeros(self.N_data)
        
        for i in range(self.N_data):
            jac = np.array([1/self.r02[i], -self.r08[i]/self.r02[i]**2])
            cov = np.zeros((2,2))
            cov[0,0] = self.r08_err[i]**2
            cov[1,1] = self.r02_err[i]**2
            
            th230_u238_act_ratio_var = jac @ cov @ jac.T
            
        return th230_u238_act_ratio, np.sqrt(th230_u238_act_ratio_var)
        
    
    def IBIS_DataFrame(self):
        if self.r48 is None:
        # If r48 is none
        # There must be a d48 so
        # we convert this into r48
            u234_u238_Act, u234_u238_Act_err = self.Convert_to_r48()
        else:
            u234_u238_Act = self.r48
            u234_u238_Act_err = self.r48_err

        if self.r28 is None:
        # If r28 is None it
        # is assumed you have input a th230/th232 measured ratio
        # we convert this as well
            if self.r02 is not None:
                th232_u238_Act, th232_u238_Act_err = self.Convert_to_r28()
            if self.u238 is not None:
                th232_u238_Act, th232_u238_Act_err = self.Convert_u_and_th_r28()
            
        else:
            th232_u238_Act = self.r28
            th232_u238_Act_err = self.r28_err
        
        df = pd.DataFrame({"Th_230_r": self.r08,
        "Th_230_r_err" : self.r08_err,
        "Th_232_r":th232_u238_Act,
        "Th_232_r_err": th232_u238_Act_err,
        "U_234_r":u234_u238_Act,
        "U_234_r_err":u234_u238_Act_err,
        
        "Depth" : self.depths,
        "Sample": self.measurement_name})
        df = df.sort_values(by = 'Depth', ascending = True)
        df.to_excel(f'/Users/jackcarter/Documents/Initial_Thorium/{self.sample}_IBIS_Input.xlsx')
