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
                int
                ).values # Convert to numpy array for consistency

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
        # Need to convert categorical variables to numerical if necessary
        self.clf.fit(
            self.df[self.variables],
            self.mask,
            )

        # Feature importance using numpy arrays
        importances = np.array(self.clf.feature_importances_)
        nonzero_indices = np.where(importances > 0)[0]
        sorted_indices = nonzero_indices[np.argsort(importances[nonzero_indices])[::-1]]
        print("Feature importances:")
        for idx in sorted_indices:
            print(f"{self.variables[idx]}: {importances[idx]}")

        # Vizualizace stromu
        plt.figure(figsize=(12, 6))
        plot_tree(
            self.clf,
            feature_names = self.variables,
            class_names = ["Not Suspect", "Suspect"],
            filled = True,
            rounded = True,
            max_depth = self.cfg.hemorrhage.model.max_depth,
        )
        plt.title("Decision Tree for Suspect Identification")
        plt.tight_layout()
        plt.show()


    def pipeline(
        self
        ) -> None:
        """_summary_
        """
        self.__load_attr___()
        self.__make_suspect__()
        self.analyze_decision_tree()

    def save_results(
        self
        ) -> None:
        """Save the results of the hemorrhage analysis."""


# Example usage
if __name__ == "__main__":
    ConfigSingleton.set()
    analysis = HemorrhageAnalysis()
    analysis.pipeline()
    analysis.save_results()
