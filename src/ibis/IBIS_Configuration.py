import numpy as np
import pandas as pd
import os


class IBIS_Configuration_Class: 
    """
    IBIS configuration class reads in raw measured activity data
    and constructs a refined dataframe for the model input, 
    plot of the input data, constructs priors for the initial thorium. 
    This is a precusor to the model and neccassary for defining parameters and 
    distributions required for the IBIS Bayesian model. 
    """
    def __init__(self, filepath, meta=None):
        """
        Initialize the IBIS model with default configurations.
        """

        self.filepath = filepath
        self.meta = dict(meta) if meta is not None else {}

        self.uncertainty_type = self.meta.get(
            "uncertainty_type",
            "abs"
        )

        self.uncert_level = float(
            self.meta.get(
                "uncertainty_level",
                1.0)
                )
        
        # Data
        self.Input_Data = None
        self.load_data()
    
    def load_data(self): 
        """
        Load measured ratios, measured ratio uncertainties, 
        depths, and depth uncertainties
        Parameters: 
        filepath (str) : Path to the file to be read in
        Returns: 
        pd.DataFrame: Dataframe contianing the loaded data
        """
        try:
            self.input_data = pd.read_excel(self.filepath)
            print(f"Data loaded successfully from {self.filepath} (Excel)")
        except Exception as e1:
            try:
                self.input_data = pd.read_csv(self.filepath)
                print(f"Data loaded successfully from {self.filepath} (CSV)")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to read '{self.filepath}' as Excel ({e1}) and as CSV ({e2})"
                )


    def ensure_data_loaded(self):
        if self.input_data is None:
            raise ValueError("Data has not been loaded. Please load data before proceeding.")
            
    def Get_Depths(self):
        self.ensure_data_loaded()
        df = self.input_data
        Depths = self._get_depth_array()
        if 'Depth_err' in df.columns:
            Depths_err = df['Depth_err'].values
        else:
            Depths_err = np.ones(Depths.size)  * 0.01 * Depths.min()  # Assume a 1% of minimum depth
        self.Depths = Depths
        self.Depths_err = Depths_err
        return self.Depths, self.Depths_err
        
    def _get_depth_array(self):
        for col in ['Depth', 'depth_top', 'depth', 'Depth_top']:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        return KeyError("Check the depth column please")


    def _230Th_val(self):
        # allowed depth column names, in order of preference
        for col in ["230Th_238U_activity", "Th_230_r", "Th230_r", "Th_230_r"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '230Th_238U_activity', 'Th_230_r', 'Th230_r').")
        
    def _230Th_err(self):
        # allowed depth column names, in order of preference
        for col in ["230Th_238U_activity_uncertainty", "Th_230_r_err", "Th230_r_err", "Th_230_r_2 SEM absolute-err"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '230Th_238U_activity_uncertainty', 'Th_230_r_err', 'Th230_r_err', 'Th_230_r_2 SEM absolute-err').")
        
    def _232Th_val(self):
        # allowed depth column names, in order of preference
        for col in ["232Th_238U_activity", "Th_232_r", "Th232_r", "Th_232_r"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '232Th_238U_activity', 'Th_232_r', 'Th232_r').")
        
    def _232Th_err(self):
        # allowed depth column names, in order of preference
        for col in ["232Th_238U_activity_uncertainty", "Th_232_r_err", "Th232_r_err", "Th_232_r_2 SEM absolute-err"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '232Th_238U_activity_uncertainty', 'Th_232_r_err', 'Th232_r_err').")
        
    def _234U_val(self):
        # allowed depth column names, in order of preference
        for col in ["234U_238U_activity", "U_234_r", "U234_r", "U_234_r"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '234U_238U_activity', 'U_234_r', 'U234_r').")
        
    def _234U_err(self):
        for col in ["234U_238U_activity_uncertainty", "U_234_r_err", "U234_r_err", "U_234_r_2 SEM absolute-err"]:
            if col in self.input_data.columns:
                return self.input_data[col].to_numpy()
        raise KeyError("No depth column found (expected one of: '234U_238U_activity_uncertainty', 'U_234_r_err', 'U234_r_err').")
        
    def ensure_depths(self):
        if self.Depths is None or self.Depths_err is None:
            self.Get_Depths()
    
    def Get_Measured_Ratios(self):
        """
        Get Data into the right formatting for uses later on within the model.
        """
        self.ensure_data_loaded()
        Depths, Depths_err = self.Get_Depths()

        # Extracting ratios and their errors
        df = self.input_data
        Th230_238U_ratios = self._230Th_val()
        Th232_238U_ratios = self._232Th_val()
        U234_U238_ratios = self._234U_val()
        Th230_238U_ratios_err = self._230Th_err()
        Th232_238U_ratios_err = self._232Th_err()
        U234_U238_ratios_err = self._234U_err()
        
        # Creaing a DataFrame with the extracted data
        # Uncertainties used throughout are at 1sigma analytical

        if self.uncertainty_type == 'abs':
            self.df_ratios = pd.DataFrame({
                "Th230_238U_ratios": Th230_238U_ratios,
                "Th230_238U_ratios_err": Th230_238U_ratios_err / self.uncert_level,
                "Th232_238U_ratios": Th232_238U_ratios,
                "Th232_238U_ratios_err": Th232_238U_ratios_err / self.uncert_level,
                "U234_U238_ratios": U234_U238_ratios,
                "U234_U238_ratios_err": U234_U238_ratios_err / self.uncert_level,
                "Depths": Depths,
                "Depths_err": Depths_err / self.uncert_level
            })
        
        elif self.uncertainty_type == "rel":
            self.df_ratios = pd.DataFrame({
                "Th230_238U_ratios": Th230_238U_ratios,
                "Th230_238U_ratios_err": (Th230_238U_ratios_err / (100 *self.uncert_level)) * Th230_238U_ratios,
                "Th232_238U_ratios": Th232_238U_ratios,
                "Th232_238U_ratios_err": (Th232_238U_ratios_err / (100 *self.uncert_level)) * Th232_238U_ratios,
                "U234_U238_ratios": U234_U238_ratios,
                "U234_U238_ratios_err": (U234_U238_ratios_err / (100 * self.uncert_level)) * U234_U238_ratios,
                "Depths": Depths,
                "Depths_err": Depths_err / self.uncert_level
            })

        return self.df_ratios

            





