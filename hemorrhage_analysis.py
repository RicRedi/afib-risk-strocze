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
# import numpy as np
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
        """Mark the hemorrhage variable as suspect."""
        _ = evaluate_logic(
            self.df,
            self.cfg.hemorrhage.conditions,
            self.cfg.hemorrhage.logic,
            )
        # Po sem to funguje...

    def pipeline(
        self
        ) -> None:
        """_summary_
        """
        self.__load_attr___()
        self.__make_suspect__()
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
