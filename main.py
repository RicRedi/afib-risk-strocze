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
This script analyzes the correlation between a set of variables
and a reference variable in a dataset.
"""
import yaml
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from core import validate_inputs_from_signature, convert_column_to_binary, remove_outliers_iqr
from plotting import CorrelationPlotter

def analyze_correlations(
    file_path: str = None,
    variables: list = None,
    reference_var: list = None,
    significance_level: float = 1.0,
    ) -> pd.DataFrame:
    """_summary_
    Args:
        file_path (_type_): _description_
        variables (_type_): _description_
        reference_var (_type_): _description_
        binary_reference (bool, optional): _description_. Defaults to False.
        significance_level (float, optional): _description_. Defaults to 0.05.
    """
    validate_inputs_from_signature(analyze_correlations, locals())

    # Load data
    df = pd.read_excel(
        file_path,
        usecols = variables + [reference_var],
        )
    results = []
    y = df[reference_var].dropna()

    for var in variables:
        print(f'Analyzing correlation for variable: {var}')

        x = df[var].apply(
            lambda x: np.nan if isinstance(x, str) else x
            ).replace(
                [np.inf, -np.inf], np.nan
                ).dropna()
        # Remove outliers using IQR method
        x = remove_outliers_iqr(x)
        # Align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        yy = convert_column_to_binary(y.loc[common_idx])

        if df[reference_var].nunique() == 2:
            # Logistic regression
            xx = sm.add_constant(x).values
            model = sm.Logit(yy, xx).fit(disp=0)
            p_value = model.pvalues[1]
            coef = model.params[1]
            corr_type = 'logistic'
        else:
            # Pearson correlation
            coef, p_value = pearsonr(x, yy)
            corr_type = 'pearson'

        results.append({
            'variable': var,
            'correlation': coef,
            'p_value': p_value,
            'type': corr_type
        })
        plotter = CorrelationPlotter(
            var=var,
            reference_var=reference_var,
            coef=coef,
            p_value=p_value
        )
        # Plot if significant
        if p_value < significance_level:
            plotter.plot(
                x=x,
                y=yy,
            )

    return pd.DataFrame(results)

# Example usage:

with open(
    'config.yaml',
    'r',
    encoding = 'utf-8',
    ) as file:
    config = yaml.safe_load(
        file,
        )

output = analyze_correlations(
    file_path = config['file_path'],
    variables = config['independent_continuous_variables'],
    reference_var = config['reference_var'],
)
# Save results to CSV
output.to_csv(
    'correlation_results.csv',
    index = False,
    encoding = 'utf-8',
)
print('Done with continuous variables analysis!')
# Dodělat......
# analyze_binary_correlations(
#     file_path = config['file_path'],
#     variables = config['independent_binary_variables'],
#     reference_var = config['reference_var'],
#     significance_level = 0.05
# )
