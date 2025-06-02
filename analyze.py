# -*- coding: utf-8 -*-

"""
Created on 02. 06. 2025 at 11:50:25

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
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import pearsonr
from core import (
    validate_inputs_from_signature,
    convert_column_to_binary,
    remove_outliers_iqr
)
from plotting import CorrelationPlotter  # assumes your plotting class is here


class VariableCorrelationAnalyzer:
    """
    Class for analyzing correlations between a set of variables and a reference variable.
    This class loads data from an Excel file, performs correlation analysis,
    and plots the results. It supports both logistic regression for binary outcomes
    and Pearson correlation for continuous outcomes.
    Attributes:
        config_path (str): Path to the configuration file.
        file_path (str): Path to the Excel file containing the data.
        variables (list): List of independent continuous variables to analyze.
        reference_var (str): The reference variable for correlation analysis.
        significance_level (float): Significance level for hypothesis testing.
        df (pd.DataFrame): DataFrame containing the loaded data.
        result (dict): Dictionary to store results of the analysis.
    """
    def __init__(
        self,
        config_path: str
        ) -> None:
        """Initializes the VariableCorrelationAnalyzer with configuration settings.
        Args:
            config_path (str): Path to the configuration file in YAML format.
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            KeyError: If required keys are missing in the configuration file.
        Returns:
            None
        """
        with open(
            config_path,
            'r',
            encoding='utf-8'
            ) as f:
            self.config = yaml.safe_load(f)

        self.variables = self.config['independent_continuous_variables']
        self.reference_var = self.config['reference_var']
        self.df = None
        self.result = {var: {} for var in self.variables}

    def __repr__(self):
        """String representation of the VariableCorrelationAnalyzer."""
        return (
            f"VariableCorrelationAnalyzer("
            f"file_path={self.config['file_path']}, "
            f"variables={self.variables}, "
            f"reference_var={self.reference_var}, "
            f"significance_level={self.config.get('significance_level', 1.0)}"
            f"save_path={self.config['save_path']}"
            )

    def pipeline(
        self,
        ) -> None:
        """ Runs the analysis pipeline: load data and analyze correlations.
        This method is a convenience method that calls load_data() and analyze().
        """
        self.load_data()
        self.analyze()
        if self.config.get('save_results', True):
            self.save_results()

    def load_data(self):
        """Loads and filters the Excel file."""
        self.df = pd.read_excel(
            self.config['file_path'],
            usecols=self.variables + [self.reference_var]
        )

    def analyze(
        self,
        ):
        """Performs correlation analysis."""
        validate_inputs_from_signature(self.analyze, {
            'self': self,
        })

        if self.df is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        y = self.df[self.reference_var].dropna()

        for var in self.variables:
            print(f'Analyzing correlation for variable: {var}')
            x = self.df[var].apply(
                lambda x: np.nan if isinstance(x, str) else x
            ).replace([np.inf, -np.inf], np.nan).dropna()

            x = remove_outliers_iqr(x)
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            yy = convert_column_to_binary(y.loc[common_idx])

            if self.df[self.reference_var].nunique() == 2:
                # Logistic regression
                xx = sm.add_constant(x).values
                model = sm.Logit(yy, xx).fit(disp=0)
                coef = model.params[1]
                p_value = model.pvalues[1]
                corr_type = 'logistic'
            else:
                coef, p_value = pearsonr(x, yy)
                corr_type = 'pearson'

            self.result[var] = {
                'correlation': coef,
                'p_value': p_value,
                'type': corr_type
            }

            plotter = CorrelationPlotter(
                var=var,
                reference_var=self.reference_var,
                coef=coef,
                p_value=p_value,
            )

            if p_value < self.config.get('significance_level', 1.0):
                plotter.plot(
                    x=x,
                    y=yy,
                    )

    def save_results(
        self,
        ) -> None:
        """Saves the analysis results to a JSON file."""
        with open(
            self.config['save_path'],
            'w',
            encoding='utf-8'
            ) as f:
            json.dump(self.result, f, ensure_ascii=False, indent=4)
