# -*- coding: utf-8 -*-

"""
Created on 03. 06. 2025 at 15:02:29

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Short description of the script.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from utils.config_singleton import ConfigSingleton
from core import load_data, cmp_tia_mapping

class TrainTestSplitter:
    """
    Class for splitting data into training and testing sets.
    This class uses the train_test_split function from sklearn to split the data
    based on the configuration settings. It can handle both continuous and categorical
    variables and can apply stratification if specified in the configuration.
    Attributes:
        cfg (Config): Configuration object containing split parameters.
    """
    def __init__(
        self,
        ) -> None:
        """Initialize the TrainTestSplitter with configuration."""
        self.cfg = ConfigSingleton.get()
        self.data = None
        self.train_set = None
        self.test_set = None
        self.df = load_data(
            self.cfg.analysis.file_path,
            self.cfg.variables.model_features,
            self.cfg.variables.reference_var
            )
        self.preprocess_data()
        self.split()

    def preprocess_data(
        self,
        ) -> None:
        """
        Preprocesses the data by removing rows with NaN values.
        This method filters the DataFrame to remove any rows that contain NaN values
        in the independent variables or the dependent variable.
        Returns:
            None
        """
        y = cmp_tia_mapping(self.df[self.cfg.variables.reference_var])
        for var in self.cfg.variables.model_features:
            self.df[var] = self.df[var].apply(
                    lambda x: np.nan if isinstance(x, str) else x
                    ).replace(
                        [np.inf, -np.inf],
                        np.nan
                        ).dropna()
            # standardize the variable using z-score normalization
            self.df[var] = (self.df[var] - self.df[var].mean()) / self.df[var].std()
        x = self.df[self.cfg.variables.model_features].values
        y = y.values
        self.data = np.hstack((x, y.reshape(-1, 1)))
        mask = ~np.isnan(self.data).any(axis=1)
        self.data = self.data[mask]

    def split(
        self,
        ) -> tuple:
        """Split the data into training and testing sets.
        Args:
            x (list): List of independent variables.
            y (list): List of dependent variables.
        Returns:
            tuple: Training and testing sets for independent and dependent variables.
        """
        self.train_set, self.test_set = train_test_split(
            self.data,
            test_size=self.cfg.model.split.test_size,
            random_state=self.cfg.model.split.random_state,
            )
    def get(
        self,
        ) -> tuple:
        """Get the training and testing sets.
        Returns:
            tuple: Training and testing sets for independent and dependent variables.
        """
        if self.train_set is None or self.test_set is None:
            raise ValueError("Data has not been split yet. Call the split method first.")
        return self.train_set, self.test_set
