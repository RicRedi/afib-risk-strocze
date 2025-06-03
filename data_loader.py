# -*- coding: utf-8 -*-

"""
Created on 03. 06. 2025 at 13:12:30

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

import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.config_singleton import ConfigSingleton

class TabularDataset(Dataset):
    """Tabular Dataset for PyTorch.
    This class is designed to load and preprocess tabular data from an Excel file.

    Args:
        Dataset (Dataset): PyTorch Dataset class for handling tabular data.
    Attributes:
        cfg (Config): Configuration object containing file paths and variable names.
        df (pd.DataFrame): DataFrame containing the loaded data.
        variables (list): List of independent variable names.
        reference_var (str): Name of the dependent variable.
    Raises:
        KeyError: If the configuration does not contain the required keys.
    """
    def __init__(
        self,
        df: pd.DataFrame = None,
        ) -> None:
        """Initializes the TabularDataset with configuration and loads data.
        This method retrieves the configuration from the ConfigSingleton,
        initializes the list of independent variables and the reference variable,
        and loads the data from the specified Excel file.
        """
        self.cfg = ConfigSingleton.get()
        self.df = df
        if self.df is None:
            raise ValueError("DataFrame cannot be None. Please provide a valid DataFrame.")
        self.x = None
        self.y = None

    def __len__(
        self,
        ) -> int:
        """
        Returns the number of samples in the dataset.
        This method returns the length of the dataset, which is the number of rows
        in the DataFrame after loading the data.
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(
        self,
        idx: int,
        ) -> tuple:
        """Returns a sample from the dataset at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the input features (X) and the target variable (y)
                   as tensors. The input features are a tensor of shape (input_size,)
                   and the target variable is a tensor of shape (output_size,).
        """
        self.y = self.df[idx,-1]
        self.x = self.df[idx,:-1]
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
