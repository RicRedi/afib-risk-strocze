# -*- coding: utf-8 -*-

"""
Created on 02. 06. 2025 at 10:31:51

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
    
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

class CorrelationPlotter:
    """
    Class for plotting correlations between variables.
    This class provides methods to plot logistic regression for binary outcomes
    and correlation for continuous outcomes. It requires the independent variable,
    dependent variable, and metadata such as variable names, coefficients, and p-values.
    Attributes:
        var (str): Name of the independent variable.
        ref (str): Name of the dependent variable.
        coef (float): Coefficient from the regression analysis.
        p_value (float): P-value from the regression analysis.
    """
    def __init__(
        self,
        var: str,
        reference_var: str,
        coef: float,
        p_value: float
        ) -> None:
        """Initialize the CorrelationPlotter with data and metadata.
        Args:
            var (str): Name of the independent variable.
            reference_var (str): Name of the dependent variable.
            coef (float): Coefficient from the regression analysis.
            p_value (float): P-value from the regression analysis.
        """
        self.var = var
        self.ref = reference_var
        self.coef = coef
        self.p_value = p_value

    def plot_logistic(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """Plot logistic regression for binary outcomes."""
        lr = LogisticRegression()
        lr.fit(x.values.reshape(-1, 1), y)
        x_plot = np.linspace(x.min(), x.max(), 100)
        y_prob = lr.predict_proba(x_plot.reshape(-1, 1))[:, 1]

        plt.plot(x_plot, y_prob, label='Logistic fit')
        plt.scatter(x, y, alpha=0.5, label='Data')
        plt.ylabel(f'{self.ref} (probability)')
        plt.xlabel(self.var)
        plt.title(f'Logistic correlation: {self.var} vs {self.ref}\n'
                  f'Coef={self.coef:.3f}, p={self.p_value:.3g}')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def plot_continuous(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """Plot correlation for continuous outcome."""
        sns.regplot(x=x, y=y, line_kws={'label': 'Regression line'})
        plt.ylabel(self.ref)
        plt.xlabel(self.var)
        plt.title(f'Continuous correlation: {self.var} vs {self.ref}\n'
                  f'Coef={self.coef:.3f}, p={self.p_value:.3g}')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def plot(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray
        ) -> None:
        """Plot the correlation based on the type of dependent variable."""
        # Validate input types and values
        self.__check__(x, y)

        plt.figure(figsize=(10, 6))
        if len(np.unique(y)) == 2:
            self.plot_logistic(x, y)
        else:
            self.plot_continuous(x, y)
        self.__add_meta_to_plot__()

    def __check__(
        self,
        x,
        y,
        ) -> None:
        """Check the validity of input data for plotting.
        Args:
            x (pd.Series | np.ndarray): Independent variable data.
            y (pd.Series | np.ndarray): Dependent variable data.
        
        Raises:
                ValueError: If any of the input data is invalid.
        """
        if not isinstance(x, (pd.Series, np.ndarray)) \
            or not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("x and y must be pandas Series or NumPy arrays.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) == 0:
            raise ValueError("x and y must not be empty.")
        if not np.issubdtype(np.array(x).dtype, np.number) \
            or not np.issubdtype(np.array(y).dtype, np.number):
            raise ValueError("x and y must contain numeric data.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("x and y must not contain NaN or infinite values.")
        if not isinstance(self.var, str) or not isinstance(self.ref, str):
            raise ValueError("var and ref must be strings representing variable names.")
        if not isinstance(self.coef, (int, float))\
            or not isinstance(self.p_value, (int, float)):
            raise ValueError("coef and p_value must be numeric values.")
        if not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be between 0 and 1.")

    def __add_meta_to_plot__(
        self
        ) -> None:
        """
        Add metadata to the plot, including variable names, coefficients, and p-values.
        This method sets the labels and title of the plot based on the initialized
        variable names, coefficient, and p-value.
        It also ensures that the plot is displayed with appropriate labels and title.
        Args:
            None
        Returns:
            None
        Raises:
            None
        
        This method is typically called after plotting the data to enhance the plot
        """
        plt.ylabel(f'{self.ref} (probability)')
        plt.xlabel(self.var)
        plt.title(f'Logistic correlation: {self.var} vs {self.ref}\n'
                  f'Coef={self.coef:.3f}, p={self.p_value:.3g}')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
