# -*- coding: utf-8 -*-

"""
Created on 30. 05. 2025 at 15:08:33

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
    Correlations Analysis

"""
from analyze import VariableCorrelationAnalyzer

analyzer = VariableCorrelationAnalyzer("config.yaml")
analyzer.pipeline()
print("done")
# Dodělat......
# analyze_binary_correlations(
#     file_path = config['file_path'],
#     variables = config['independent_binary_variables'],
#     reference_var = config['reference_var'],
#     significance_level = 0.05
# )
