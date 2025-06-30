# -*- coding: utf-8 -*-

"""
Created on 23. 06. 2025 at 10:33:54

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
import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils.config_singleton import ConfigSingleton
from plotting import CorrelationPlotter#, DecisionTreePlotter
from core import (
    load_data,
    evaluate_logic,
    remove_outliers_iqr_df,
)

class HemorrhageAnalysis:
    """Class for analyzing hemorrhage data.
    This class provides methods to analyze hemorrhage data based on the configuration settings.
    It includes methods to run the analysis and save the results.
    """

    def __init__(
        self,
        ) -> None:
        """Initialize the HemorrhageAnalysis with configuration settings."""
        self.cfg = ConfigSingleton.get()
        self.df = None
        self.result = {}
        self.mask = None
        self.clf = None
        self.x = None

    def __load_attr___(
        self,
        variables_key: str = "variables",
        ) -> None:
        """Load attributes from the configuration settings."""
        if not self.cfg.hemorrhage.variables or not self.cfg.hemorrhage.reference_var:
            raise KeyError(
                f"Configuration must contain {variables_key} "
                "and 'reference_var'."
            )
        self.df = load_data(
            self.cfg.analysis.file_path,
            self.cfg.hemorrhage.variables,
            self.cfg.hemorrhage.reference_var,
            )
        self.result = {var: {} for var in self.cfg.hemorrhage.variables}
    def __make_suspect__(
        self,
        ) -> None:
        """
        Mark the hemorrhage variable as suspect.
        Defined in the configuration file.
        This method evaluates the conditions and logic defined in the configuration
        """
        self.mask = evaluate_logic(
            self.df,
            self.cfg.hemorrhage.conditions,
            self.cfg.hemorrhage.logic,
            ).astype(    # Convert the mask to boolean
                int,
                ) # Keep series type for consistency with the DataFrame

    def __variable_preprocessing__(
        self,
        ) -> np.ndarray:
        """Preprocess the variables for analysis.
        This method can include normalization, encoding, or other preprocessing steps
        as defined in the configuration.
        """
        self.x = self.df[self.cfg.hemorrhage.variables]
        self.x = self.x.map(
            lambda x: np.nan if isinstance(x, str) else x
            ).replace(
                [np.inf, -np.inf],
                np.nan
                ).dropna()
        self.x = remove_outliers_iqr_df(
                self.x,
                threshold = self.cfg.analysis.iqr_threshold,
                )
        common_idx = self.x.index.intersection(self.mask.index)
        self.x = self.x.loc[common_idx]
        yy = self.mask.loc[common_idx]
        return yy

    def analyze_decision_tree(
        self,
        ) -> None:
        """
        Analyze the decision tree for hemorrhage detection.
        """
        # x = self.df[self.cfg.hemorrhage.variables]
        # y = self.mask
        if self.mask is None:
            raise ValueError("Mask is not defined. Please run __make_suspect__() first.")

        self.clf = DecisionTreeClassifier(
            max_depth = self.cfg.hemorrhage.model.max_depth,
            min_samples_split = self.cfg.hemorrhage.model.min_samples_split,
            min_samples_leaf = self.cfg.hemorrhage.model.min_samples_leaf,
            random_state = self.cfg.hemorrhage.model.random_state,
            class_weight = 'balanced',
        )

        # Preprocess the variables
        y = self.__variable_preprocessing__()

        self.clf.fit(
            self.x.values,
            y.values,
            )

        # Feature importance using numpy arrays
        importances = np.array(self.clf.feature_importances_)
        for var in self.cfg.hemorrhage.variables:
            self.result[var]["importance"] = importances[self.cfg.hemorrhage.variables.index(var)]
        nonzero_indices = np.where(importances > 0)[0]
        sorted_indices = nonzero_indices[np.argsort(importances[nonzero_indices])[::-1]]
        cms = np.cumsum(importances[sorted_indices])
        for i, c in enumerate(cms):
            if c < 0.6:
                xx = sm.add_constant(
                    self.x[self.cfg.hemorrhage.variables[sorted_indices[i]]],
                )
                model = sm.Logit(y, xx).fit(disp=0)
                CorrelationPlotter(
                    var = self.cfg.hemorrhage.variables[sorted_indices[0]],
                    reference_var = "Suspect",
                    coef = model.params[1],  # Coefficient for the variable
                    p_value = model.pvalues[1],  # Placeholder for p-value
                ).plot(
                    self.x[self.cfg.hemorrhage.variables[sorted_indices[i]]],
                    y,
                )
        # print("Feature importances:")
        # for idx in sorted_indices:
        #     print(f"{self.cfg.hemorrhage.variables[idx]}: {importances[idx]}")

        # Vizualizace stromu
        plt.figure(
            figsize = (
                self.cfg.hemorrhage.tree_plot.width,
                self.cfg.hemorrhage.tree_plot.height,
                )
            )
        plot_tree(
            self.clf,
            feature_names = self.cfg.hemorrhage.variables,
            class_names = ["Not Suspect", "Suspect"],
            filled = True,
            rounded = True,
            max_depth = self.cfg.hemorrhage.model.max_depth,
            fontsize = self.cfg.hemorrhage.tree_plot.fontsize,
        )
        plt.title("Decision Tree for Suspect Identification")
        plt.tight_layout()
        

    def pipeline(
        self
        ) -> None:
        """
        Run the analysis pipeline for hemorrhage detection.
        This method orchestrates the loading of attributes, making the suspect variable,
        and analyzing the decision tree.
        It prints a message indicating that the analysis is complete
        and results are ready for saving.
        """
        self.__load_attr___()
        self.__make_suspect__()
        self.analyze_decision_tree()
        if self.cfg.hemorrhage.output.save:
            self.save_results()

    def save_results(
        self
        ) -> None:
        """Save the results of the hemorrhage analysis."""
        # save results as a json file
        os.makedirs(
            os.path.dirname(
                self.cfg.hemorrhage.output.save_path,
                ),
            exist_ok = True,
            )
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filename = f"importances_{timestamp}.json"
        with open(
            os.path.join(
            self.cfg.hemorrhage.output.save_path,
            filename
            ),
            'w',
            encoding = 'utf-8',
        ) as f:
            json.dump(
            self.result,
            f,
            indent = 4,
            ensure_ascii = False  # Allows non-ASCII characters like French and Czech apostrophes
            )

# Example usage
if __name__ == "__main__":
    ConfigSingleton.set()
    analysis = HemorrhageAnalysis()
    analysis.pipeline()
