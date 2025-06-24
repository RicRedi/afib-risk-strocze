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
    Module containing utility functions for data validation and preprocessing.
    This module includes functions to validate function inputs, convert binary columns,
    and remove outliers using IQR and Z-score methods.
"""
import inspect
from typing import Callable, Dict
import operator
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

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

def load_data(
    path: str,
    variables: list,
    reference_var: str|list,
    ) -> None:
    """Loads and filters the Excel file."""
    # if list, usecols = variables + reference_var (list)
    # if str, usecols = variables + [reference_var]
    if isinstance(reference_var, str):
        reference_var = [reference_var]
    elif not isinstance(reference_var, list):
        raise TypeError("reference_var must be a string or a list of strings.")
    if not isinstance(variables, list):
        raise TypeError("variables must be a list of strings.")
    return pd.read_excel(
        path,
        usecols = variables + reference_var
        )

def convert_column_to_binary(
    series: pd.Series,
    numpy: bool = True,
    ) -> pd.Series| np.ndarray:
    """Converts a pandas Series of textual binary values to a NumPy array of binary integers (0/1).
    The function normalizes string values (trimming whitespace and converting to lowercase)
    and maps common binary representations (e.g., 'yes', 'no', 'true', 'false', '1', '0', etc.)
    to their corresponding integer values. Returns a NumPy array of 0s and 1s.
    
    Args:
        series (pd.Series): Input pandas Series containing textual binary values.
        
    Returns:
        _ (np.ndarray): NumPy array of binary integers (0 or 1).
    """
    if numpy:
        return series.str.strip().str.lower().map({
            'yes': 1, 'y': 1, 'true': 1, '1': 1, 'checked': 1, 'ano': 1,
            'no': 0, 'n': 0, 'false': 0, '0': 0, 'unchecked': 0, 'ne': 0,
        }).values

    return series.str.strip().str.lower().map({
            'yes': 1, 'y': 1, 'true': 1, '1': 1, 'checked': 1, 'ano': 1,
            'no': 0, 'n': 0, 'false': 0, '0': 0, 'unchecked': 0, 'ne': 0,
        })

def cmp_tia_mapping(
    x: pd.Series,
    ) -> pd.Series:
    """Maps values in a pandas Series to integers based on a provided mapping dictionary.
    The function replaces each value in the Series with its corresponding integer from the mapping.
    If a value is not found in the mapping, it is replaced with NaN.
    
    Args:
        x (pd.Series): Input pandas Series containing values to be mapped.
        
    Returns:
        pd.Series: A new pandas Series with values replaced by their corresponding integers
                   from the mapping.
    """
    top_two = x.value_counts().index[:2]
    mapping = {top_two[0]: 1, top_two[1]: 0}
    return x.map(mapping)

def remove_outliers_iqr(
    x: pd.Series,
    threshold: float = 1.5,
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
    threshold: float = 3.0,
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

def make_condition(
    df: pd.DataFrame,
    cond: dict
    ) -> pd.Series:
    """
    Creates a boolean mask for a DataFrame based on a condition.
    The condition is specified as a dictionary with keys 'col', 'op', and 'value'.
    The 'col' key specifies the column to apply the condition to,
    the 'op' key specifies the operator (e.g., '==', '!=', '<=', '>=', '<', '>', 'in', 'not in'),
    and the 'value' key specifies the value to compare against.
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        cond (dict): A dictionary specifying the condition with keys 'col', 'op', and 'value'.      
    
    Returns:
        pd.Series: A boolean Series indicating which rows in the DataFrame satisfy the condition.
    
    Raises:     
        ValueError: If the operator is not supported.
        """
    col, op, val = cond.col, cond.op, cond.value
    ops = {
        '==': operator.eq,
        '!=': operator.ne,
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
        'in': lambda x, y: x.isin(y),
        'not in': lambda x, y: ~x.isin(y),
    }
    if ptypes.is_string_dtype(df[col]):
        df[col] = convert_column_to_binary(df[col], numpy=False)
    try:
        return ops[op](df[col], val)
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {op}") from exc

def evaluate_logic(
    df,
    conditions: Dict[str, dict],
    logic: str
    ) -> pd.Series:
    """_summary_

    Args:
        df (_type_): _description_
        conditions (Dict[str, dict]): _description_
        logic (str): _description_

    Returns:
        pd.Series: _description_
    """
    name_list = list(vars(conditions).keys())
    local_vars = {
        name: make_condition(df, getattr(conditions, name)) \
            for _, name in enumerate(name_list)
        }
    return eval(logic, {"__builtins__": {}}, local_vars)
