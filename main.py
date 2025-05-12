""" Correlations Analysis
This script analyzes the correlation between a set of variables
and a reference variable in a dataset.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def analyze_correlations(
    file_path,
    variables,
    reference_var,
    binary_reference=False,
    significance_level=0.05
    ):
    """_summary_
    Args:
        file_path (_type_): _description_
        variables (_type_): _description_
        reference_var (_type_): _description_
        binary_reference (bool, optional): _description_. Defaults to False.
        significance_level (float, optional): _description_. Defaults to 0.05.
    """
    # Load data
    df = pd.read_csv(file_path)
    results = []

    for var in variables:
        if var == reference_var:
            continue

        x = df[var].dropna()
        y = df[reference_var].dropna()

        # Align indices
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        if binary_reference or df[reference_var].nunique() == 2:
            # Logistic regression
            xx = sm.add_constant(x)
            model = sm.Logit(y, xx).fit(disp=0)
            p_value = model.pvalues[1]
            coef = model.params[1]
            corr_type = 'logistic'
        else:
            # Pearson correlation
            coef, p_value = pearsonr(x, y)
            corr_type = 'pearson'

        results.append({
            'variable': var,
            'correlation': coef,
            'p_value': p_value,
            'type': corr_type
        })

        # Plot if significant
        if p_value < significance_level:
            plt.figure()
            if corr_type == 'logistic':
                # Fit logistic regression for plot
                lr = LogisticRegression()
                lr.fit(x.values.reshape(-1, 1), y)
                x_plot = np.linspace(x.min(), x.max(), 100)
                y_prob = lr.predict_proba(x_plot.reshape(-1, 1))[:, 1]
                plt.plot(x_plot, y_prob, label='Logistic fit')
                plt.scatter(x, y, alpha=0.5, label='Data')
                plt.ylabel(f'{reference_var} (probability)')
            else:
                sns.regplot(x=x, y=y, logistic=False, ci=None)
                plt.ylabel(reference_var)
            plt.xlabel(var)
            plt.title(f'{corr_type.capitalize()} correlation: {var} vs {reference_var}\n'
                      f'Coef={coef:.3f}, p={p_value:.3g}')
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Print summary
    print(pd.DataFrame(results))

# Example usage:
# analyze_correlations(
#     file_path='your_table.csv',
#     variables=['age', 'bmi', 'bp', 'cholesterol'],
#     reference_var='outcome',
#     binary_reference=True
# )
