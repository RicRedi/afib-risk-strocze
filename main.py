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
from hemorrhage_analysis import HemorrhageAnalysis
from utils.config_singleton import ConfigSingleton

i = 1
ConfigSingleton.set()
if i == 1:
    VariableCorrelationAnalyzer().pipeline()
else:
    HemorrhageAnalysis().pipeline()
