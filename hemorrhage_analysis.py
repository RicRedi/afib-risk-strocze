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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils.config_singleton import ConfigSingleton
from core import (
    load_data,
    evaluate_logic,
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
        self.variables = []
        self.reference_var = None
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
        self.variables = getattr(
            self.cfg.hemorrhage,
            variables_key,
            [],
            )
        self.reference_var = self.cfg.hemorrhage.reference_var
        if not self.variables or not self.reference_var:
            raise KeyError(
                f"Configuration must contain {variables_key} "
                "and 'reference_var'."
            )
        self.df = load_data(
            self.cfg.analysis.file_path,
            self.variables,
            self.reference_var,
            )
        self.result = {var: {} for var in self.variables}
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

    def analyze_decision_tree(
        self,
        ) -> None:
        """
        Analyze the decision tree for hemorrhage detection.
        """
        # x = self.df[self.variables]
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
            self.x,
            y,
            )

        # Feature importance using numpy arrays
        importances = np.array(self.clf.feature_importances_)
        for var in self.variables:
            self.result[var]["importance"] = importances[self.variables.index(var)]
        # nonzero_indices = np.where(importances > 0)[0]
        # sorted_indices = nonzero_indices[np.argsort(importances[nonzero_indices])[::-1]]
        # print("Feature importances:")
        # for idx in sorted_indices:
        #     print(f"{self.variables[idx]}: {importances[idx]}")

        # Vizualizace stromu
        plt.figure(
            figsize = (
                self.cfg.hemorrhage.tree_plot.width,
                self.cfg.hemorrhage.tree_plot.height,
                )
            )
        plot_tree(
            self.clf,
            feature_names = self.variables,
            class_names = ["Not Suspect", "Suspect"],
            filled = True,
            rounded = True,
            max_depth = self.cfg.hemorrhage.model.max_depth,
            fontsize = self.cfg.hemorrhage.tree_plot.fontsize,
        )
        plt.title("Decision Tree for Suspect Identification")
        plt.tight_layout()
        if self.cfg.hemorrhage.tree_plot.save:
            os.makedirs(self.cfg.hemorrhage.tree_plot.save_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    self.cfg.hemorrhage.tree_plot.save_path,
                    "decision_tree.png"
                ),
                dpi = self.cfg.hemorrhage.tree_plot.dpi,
            )
        else:
            print("Tree plot will not be saved as per configuration.")
            plt.show(block = False)

    def __variable_preprocessing__(
        self,
        ) -> np.ndarray:
        """Preprocess the variables for analysis.
        This method can include normalization, encoding, or other preprocessing steps
        as defined in the configuration.
        """
        self.x = self.df[self.variables]
        self.x = self.x.applymap(
            lambda x: np.nan if isinstance(x, str) else x
            ).replace(
                [np.inf, -np.inf],
                np.nan
                ).dropna()
        common_idx = self.x.index.intersection(self.mask.index)
        self.x = self.x.loc[common_idx].values
        yy = self.mask.loc[common_idx].values
        return yy

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
        with open(
            os.path.join(
                self.cfg.hemorrhage.output.save_path,
                "importances.json"
                ),
            'w',
            encoding =  'utf-8',
            ) as f:
            json.dump(
                self.result,
                f,
                indent = 4,
                )

# Example usage
if __name__ == "__main__":
    ConfigSingleton.set()
    analysis = HemorrhageAnalysis()
    analysis.pipeline()
