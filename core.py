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
import ast
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
            'yes': 1, 'y': 1, 'true': 1, '1': 1, 'checked': 1, 'ano': 1, 'muž': 1, 'žena': 0,
            'no': 0, 'n': 0, 'false': 0, '0': 0, 'unchecked': 0, 'ne': 0,
        }).values

    return series.str.strip().str.lower().map({
            'yes': 1, 'y': 1, 'true': 1, '1': 1, 'checked': 1, 'ano': 1, 'muž': 1, 'žena': 0,
            'no': 0, 'n': 0, 'false': 0, '0': 0, 'unchecked': 0, 'ne': 0,
        })

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
    return x[(x >= lower_bound) & (x <= upper_bound)]

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
        df[col] = convert_column_to_binary(
            df[col],
            numpy = False,
            )
    try:
        return ops[op](
            df[col],
            val,
            )
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {op}") from exc

def _evaluate_ast(
    node,
    local_vars: Dict
    ) -> pd.Series:
    """
    Recursively evaluates an AST node for boolean logic operations.
    Supports 'and', 'or', and 'not' operations on pandas Series.
    
    Args:
        node: AST node to evaluate.
        local_vars (Dict): Dictionary mapping variable names to pandas Series (boolean masks).
    
    Returns:
        pd.Series: Result of the evaluation as boolean Series.
    
    Raises:
        ValueError: If unsupported operation is encountered or unknown variable referenced.
    """
    if isinstance(node, ast.BoolOp):
        # Handle 'and' and 'or' operations
        values = [_evaluate_ast(v, local_vars) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for val in values[1:]:
                result = result & val
            return result
        elif isinstance(node.op, ast.Or):
            result = values[0]
            for val in values[1:]:
                result = result | val
            return result
    elif isinstance(node, ast.Name):
        # Handle variable names (condition references)
        if node.id not in local_vars:
            raise ValueError(f"Unknown variable: {node.id}")
        return local_vars[node.id]
    elif isinstance(node, ast.UnaryOp):
        # Handle 'not' operation
        operand = _evaluate_ast(node.operand, local_vars)
        if isinstance(node.op, ast.Not):
            return ~operand
        else:
            raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
    else:
        raise ValueError(f"Unsupported operation: {type(node).__name__}")

def evaluate_logic(
    df: pd.DataFrame,
    conditions: Dict[str, dict],
    logic: str
    ) -> pd.Series:
    """
    Safely evaluates a boolean logic expression on DataFrame conditions using AST parsing.
    
    Evaluates expressions like 'cond1 and cond2' or '(cond1 or cond2) and not cond3'
    without using eval(), preventing potential code injection vulnerabilities.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        conditions (Dict[str, dict]): Dictionary of conditions with keys mapping
        to condition specifications.
        logic (str): Boolean logic expression using condition names and operators (and, or, not).
                     Example: 'condition1 and (condition2 or not condition3)'
    
    Returns:
        pd.Series: Boolean Series indicating rows that satisfy the logic expression.
    
    Raises:
        ValueError: If the logic expression contains unsupported operations or unknown variables.
        SyntaxError: If the logic expression has syntax errors.
    """
    # Build local variables mapping condition names to boolean masks
    name_list = list(vars(conditions).keys())
    local_vars = {
        name: make_condition(df, getattr(conditions, name))
        for name in name_list
    }

    # Parse the logic string into an Abstract Syntax Tree
    try:
        tree = ast.parse(logic, mode='eval')
    except SyntaxError as exc:
        raise SyntaxError(f"Invalid logic expression: {logic}") from exc

    # Evaluate the AST safely without using eval()
    return _evaluate_ast(tree.body, local_vars)
