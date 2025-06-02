# -*- coding: utf-8 -*-

"""
Created on 30. 05. 2025 at 15:18:43

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
    
    Last modified on 01. 06. 2025 at 18:43:59
"""
import inspect
from typing import Callable, Dict
import pandas as pd

def validate_inputs_from_signature(
    func: Callable = None,
    locals_dict: Dict = None,
    ) -> None:
    """
    Validates that all required arguments from a function's signature are present
    in the provided locals dictionary.
    
    Args:
        func (callable): The function whose signature will be validated.
        locals_dict (dict): A dictionary representing the local variables
                            to check for required arguments.

    Raises:
        ValueError: If any required argument (without a default value) is missing from locals_dict.
    """
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and name not in locals_dict:
            raise ValueError(f"Missing required argument: {name}")

def convert_column_to_binary(
    series: pd.Series
    ) -> pd.Series:
    """Converts a pandas Series of textual binary values to a NumPy array of binary integers (0/1).
    The function normalizes string values (trimming whitespace and converting to lowercase)
    and maps common binary representations (e.g., 'yes', 'no', 'true', 'false', '1', '0', etc.)
    to their corresponding integer values. Returns a NumPy array of 0s and 1s.
    
    Args:
        series (pd.Series): Input pandas Series containing textual binary values.
        np.ndarray: NumPy array of binary integers (0 or 1).
        
    Returns:
        _ (np.ndarray): NumPy array of binary integers (0 or 1).
    """
    return series.str.strip().str.lower().map({
        'yes': 1, 'y': 1, 'true': 1, '1': 1, 'Checked': 1, 'Ano': 1,
        'no': 0, 'n': 0, 'false': 0, '0': 0, 'Unchecked': 0, 'Ne': 0,
    }).values

def remove_outliers_iqr(
    x: pd.Series,
    threshold: float = 1.5
    ) -> pd.Series:
    """Removes outliers from a pandas Series using the Interquartile Range (IQR) method.
    The function calculates the first (Q1) and third quartiles (Q3) of the data,
    computes the IQR, and identifies outliers as values that fall below Q1 - threshold * IQR
    or above Q3 + threshold * IQR. It returns a new Series with outliers removed.
    Args:
        x (pd.Series): Input pandas Series from which to remove outliers.
        threshold (float, optional): Multiplier for the IQR to define outliers. Defaults to 1.5.
    Returns:
        pd.Series: A new pandas Series with outliers removed.
    """
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return x[(x >= lower_bound) & (x <= upper_bound)].reset_index(drop=True)

def remove_outliers_z_score(
    x: pd.Series,
    threshold: float = 3.0
    ) -> pd.Series:
    """Removes outliers from a pandas Series using the Z-score method.
    The function calculates the mean and standard deviation of the data,
    computes the Z-scores, and identifies outliers as values with an absolute Z-score
    greater than the specified threshold. It returns a new Series with outliers removed.
    
    Args:
        x (pd.Series): Input pandas Series from which to remove outliers.
        threshold (float, optional): Z-score threshold to define outliers. Defaults to 3.0.
        
    Returns:
        pd.Series: A new pandas Series with outliers removed.
    """
    mean = x.mean()
    std_dev = x.std()
    z_scores = (x - mean) / std_dev
    return x[abs(z_scores) <= threshold].reset_index(drop=True)
