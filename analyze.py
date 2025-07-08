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
    Module for analyzing correlations between a set of variables and a reference variable.
    This module supports both logistic regression for binary outcomes and Pearson correlation
    for continuous outcomes. It loads data from an Excel file, performs the analysis,
    and plots the results.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, ttest_ind, levene
from scipy.stats import chi2_contingency#, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from core import (
    validate_inputs_from_signature,
    convert_column_to_binary,
    remove_outliers_iqr,
    load_data,
    evaluate_logic,
    )
from plotting import CorrelationPlotter  # assumes your plotting class is here
from utils.config_singleton import ConfigSingleton


class VariableCorrelationAnalyzer:
    """
    Class for analyzing correlations between a set of variables and a reference variable.
    This class loads data from an Excel file, performs correlation analysis,
    and plots the results. It supports both logistic regression for binary outcomes
    and Pearson correlation for continuous outcomes.
    Attributes:
        config_path (str): Path to the configuration file.
        variables (list): List of independent continuous variables to analyze.
        reference_var (str): The reference variable for correlation analysis.
        df (pd.DataFrame): DataFrame containing the loaded data.
        result (dict): Dictionary to store results of the analysis.
    """
    def __init__(
        self,
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
        self.cfg = ConfigSingleton.get()
        self.variables = []
        self.df = None
        self.result = {}

    def __reset__(
        self,
        ) -> None:
        """Resets the analyzer's state by clearing the DataFrame and results."""
        self.variables = []
        self.df = None
        self.result = {}

    def __repr__(
        self,
        ) -> str:
        """String representation of the VariableCorrelationAnalyzer."""
        return (
            f"VariableCorrelationAnalyzer("
            f"file_path = {self.cfg.analysis.file_path}, "
            f"variables = {self.variables}, "
            f"reference_var = {self.cfg.variables.reference_var}, "
            f"significance_level = {self.cfg.analysis.significance_level}, "
            f"save_path = {self.cfg.analysis.save_path}"
            )

    def __load_attr__(
        self,
        variables_key: str = 'independent_continuous_variables',
        ) -> None:
        """Loads the configuration from the YAML file.
        Args:
            variables_key (str): Key in the configuration file that contains
                                 the list of independent variables.
        Raises:
            KeyError: If the specified key is not found in the configuration.
        Returns:
            None
        """
        self.variables = self.cfg.hemorrhage.variables
        #        getattr(self.cfg.variables, variables_key, [])
        # self.reference_var = self.cfg.variables.reference_var
        if not self.variables or not self.cfg.variables.reference_var:
            raise KeyError(
                f"Configuration must contain {variables_key} "
                "and 'reference_var'."
            )
        self.df = load_data(
            self.cfg.analysis.file_path,
            # self.variables,
            # self.cfg.variables.reference_var,
            self.cfg.hemorrhage.variables,
            self.cfg.hemorrhage.reference_var,
            )
        self.result = {var: {} for var in self.variables}

    def pipeline(
        self,
        ) -> None:
        """
        Runs the analysis pipeline: load data and analyze correlations.
        This method is a convenience method that calls load_data() and analyze().
        """
        self.analyze_continuous()
        self.analyze_binary()

    def analyze_continuous(
        self,
        ) -> None:
        """Performs correlation analysis."""
        validate_inputs_from_signature(self.analyze_continuous, {
            'self': self,
        })
        self.__reset__()
        self.__load_attr__(
            variables_key='independent_continuous_variables'
            )
        if self.df is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        # y = cmp_tia_mapping(
        #     self.df[self.cfg.variables.reference_var],
        #     ).dropna()

        y = evaluate_logic(
            self.df,
            self.cfg.hemorrhage.conditions,
            self.cfg.hemorrhage.logic,
            ).astype(    # Convert the mask to boolean
                int,
                )

        for var in self.variables:
            if self.cfg.analysis.print_progress:
                print(
                    f'Analyzing correlation for: {var} vs {self.cfg.variables.reference_var}'
                    )
            x = self.df[var].apply(
                lambda x: np.nan if isinstance(x, str) else x
                ).replace(
                    [np.inf, -np.inf],
                    np.nan
                    ).dropna()

            x = remove_outliers_iqr(
                x,
                threshold = self.cfg.analysis.iqr_threshold,
                )
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            yy = y.loc[common_idx].values

            if len(
                np.unique(yy)
                ) == 2:
                # Logistic regression
                coef, p_value = self._logistic_regression(
                    x,
                    yy
                    )
                corr_type = 'logistic'
                # check equality of variance

                # Split x by binary outcome
                group0 = x[yy == 0]
                group1 = x[yy == 1]

                # Check if both groups have enough samples
                if len(group0) > 1 and len(group1) > 1:
                    # Test for equality of variances
                    _, p_var = levene(
                        group0,
                        group1,
                        )
                    equal_var = p_var > 0.05

                    # t-test with appropriate variance setting
                    t_stat, t_p_value = ttest_ind(
                        group0,
                        group1,
                        equal_var = equal_var,
                        nan_policy = 'omit'
                        )
                else:
                    t_stat, t_p_value, p_var = np.nan, np.nan, np.nan
            else:
                coef, p_value = pearsonr(
                    x,
                    yy,
                    )
                corr_type = 'pearson'
                t_stat, t_p_value, p_var = np.nan, np.nan, np.nan


            self.result[var] = {
                'correlation': coef,
                'p_value': p_value,
                'type': corr_type,
                'data used [%]': np.round(len(x) / len(self.df[var]) * 100, 2),
                't_statistic': t_stat,
                't_p_value': t_p_value,
            }

            CorrelationPlotter(
                var = var,
                reference_var = self.cfg.variables.reference_var,
                coef = coef,
                p_value = p_value,
                ).plot(
                    x = x,
                    y = yy,
                    )
        # Save results if configured to do so
        if self.cfg.analysis.save_results:
            self.save_results(
                variables_key = 'continuous_variables_cmp_tia_mapping',
                )

    def _logistic_regression(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> tuple[float, float]:
        """Performs logistic regression and returns the coefficient and p-value."""
        xx = sm.add_constant(x).values
        model = sm.Logit(y, xx).fit(disp=0)
        return model.params[1], model.pvalues[1]

    def analyze_binary(
        self,
        ) -> None:
        """Analyzes binary correlations."""
        validate_inputs_from_signature(self.analyze_binary, {
            'self': self,
        })
        self.__reset__()
        self.__load_attr__(
            variables_key = 'independent_binary_variables'
            )
        if self.df is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        # y = cmp_tia_mapping(
        #     self.df[self.cfg.variables.reference_var]
        #     ).dropna()

        y = evaluate_logic(
            self.df,
            self.cfg.hemorrhage.conditions,
            self.cfg.hemorrhage.logic,
            ).astype(    # Convert the mask to boolean
                int,
                )

        for var in self.variables:
            if self.cfg.analysis.print_progress:
                print(f'Analyzing binary association: {var} vs {self.cfg.variables.reference_var}')
            x = self.df[var].apply(
                lambda x: np.nan if x == 'Nezjištěno' else x
                ).dropna()

            # Ensure binary
            if x.nunique() != 2 or y.nunique() != 2:
                print(f"Skipping {var} - not binary.")
                continue

            # Align x and y
            common_idx = x.index.intersection(y.index)
            x = convert_column_to_binary(x.loc[common_idx])
            yy = y.loc[common_idx].values

            # Contingency table
            contingency = pd.crosstab(x, yy)

            if contingency.shape != (2, 2):
                print(f"Skipping {var} - incomplete 2x2 table.")
                continue

            # Chi-squared test
            chi2, p_value, _, _ = chi2_contingency(contingency)
            CorrelationPlotter(
                var = var,
                reference_var = self.cfg.variables.reference_var,
                coef = chi2,
                p_value = p_value,
                ).plot(
                    x,
                    yy,
                    )

            # Odds ratio + CI
            table = Table2x2(contingency.values)
            odds_ratio = table.oddsratio
            confint = table.oddsratio_confint()

            self.result[var] = {
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'odds_CI_lower': confint[0],
                'odds_CI_upper': confint[1],
                'type': 'binary-association',
                'data used [%]': np.round(len(x) / len(self.df[var]) * 100, 2),
            }

        # Save results if configured to do so
        if self.cfg.analysis.save_results:
            self.save_results(
                variables_key = 'binary_variables_cmp_tia_mapping',
                )

    def save_results(
        self,
        variables_key: str = 'continuous_variables',
        ) -> None:
        """
        Saves the analysis results to a JSON file.
        Args:
            variables_key (str): Key to identify the type of variables in the results.
        Raises:
            KeyError: If the 'save_path' key is not found in the configuration.
        Returns:
            None
        """
        with open(
            self.cfg.analysis.save_path + \
                f'/analysis_results_{variables_key}_correlations_hemorrhage.json',
            'w',
            encoding = 'utf-8'
            ) as f:
            json.dump(
                self.result,
                f,
                ensure_ascii = False,
                indent = 4
                )

# Example usage:
# if __name__ == "__main__":
#     analyzer = VariableCorrelationAnalyzer()
#     analyzer.pipeline()
