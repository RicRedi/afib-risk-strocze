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
    Script for plotting correlations between variables.
"""
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils.config_singleton import ConfigSingleton

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
        p_value: float,
        ) -> None:
        """Initialize the CorrelationPlotter with data and metadata.
        Args:
            var (str): Name of the independent variable.
            reference_var (str): Name of the dependent variable.
            coef (float): Coefficient from the regression analysis.
            p_value (float): P-value from the regression analysis.
        """
        self.cfg = ConfigSingleton.get()
        self.var = var
        self.ref = reference_var
        self.coef = coef
        self.p_value = p_value
        # make a dir for plots if it does not exist
        if self.cfg.plotting.save_plots:
            os.makedirs('./results/plots', exist_ok=True)

    def plot_logistic(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """Plot logistic regression for binary outcomes."""
        lr = LogisticRegression()
        lr.fit(x.values.reshape(-1, 1), y)
        x_plot = np.linspace(
            x.min(),
            x.max(),
            100,
            )
        y_prob = lr.predict_proba(x_plot.reshape(-1, 1))[:, 1]

        plt.plot(
            x_plot,
            y_prob,
            label='Logistic fit',
            )
        plt.scatter(
            x,
            y,
            alpha=0.5,
            label='Data',
            )
        plt.legend()

    def plot_continuous(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """Plot correlation for continuous outcome."""
        sns.regplot(
            x = x,
            y = y,
            line_kws={'label': 'Regression line'},
            )
        plt.legend()

    def plot_binary_heatmap(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """Plot binary bars for binary independent variable."""
        # Create 2x2 contingency table
        contingency = pd.crosstab(
            x,
            y,
            )
        # Plot heatmap
        sns.heatmap(
            contingency,
            annot = True,
            fmt = 'd',
            cmap = 'Blues',
            cbar = False
        )

    def plot(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        ) -> None:
        """
        Plot the correlation based on the type of dependent variable.
        Args:
            x (pd.Series | np.ndarray): Independent variable data.
            y (pd.Series | np.ndarray): Dependent variable data.
        Returns:
            None
        """
        # Validate input types and values
        self.__check__(x, y)

        plt.figure(
            figsize = (
                self.cfg.plotting.figsize.width,
                self.cfg.plotting.figsize.height
                )
            )
        if len(np.unique(y)) == 2 and len(np.unique(x)) == 2:
            self.plot_binary_heatmap(x,y)
        elif len(np.unique(y)) == 2:
            self.plot_logistic(x, y)
        else:
            self.plot_continuous(x, y)
        self.__add_meta_to_plot__()

    def __check__(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
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
        self,
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
        plt.xticks(
            fontsize = self.cfg.plotting.fontsize.tick,
            )
        plt.yticks(
            fontsize = self.cfg.plotting.fontsize.tick,
            )
        plt.ylabel(
            f'{self.ref}',
            fontsize = self.cfg.plotting.fontsize.ylabel,
            )
        plt.xlabel(
            self.var,
            fontsize = self.cfg.plotting.fontsize.xlabel,
            )
        plt.title(f'Logistic correlation: {self.var} vs {self.ref}\n'
                  f'Coef = {self.coef:.3f}, p = {self.p_value:.3g}',
                  fontsize = self.cfg.plotting.fontsize.title,
                  )
        plt.tight_layout()
        if self.cfg.plotting.save_plots:
            var_safe = self._sanitize_filename_part(self.var)
            ref_safe = self._sanitize_filename_part(self.ref)
            try:
                plt.savefig(
                    self.cfg.plotting.save_path + f'/{var_safe}_vs_{ref_safe}.png',
                    dpi = self.cfg.plotting.dpi,
                    bbox_inches = 'tight',
                    )
            except OSError as e:
                print(f"Error saving plot: {e}")
        if self.cfg.plotting.show_plots:
            plt.show(block = self.cfg.plotting.block_plots)

    def _sanitize_filename_part(
        self,
        s: str,
        ) -> str:
        """
        Sanitize a string to be used as a part of a filename.
        This function replaces any character that is not alphanumeric, underscore, or hyphen
        with an underscore. It is useful for creating valid filenames from variable names.
        Args:
            s (str): The string to sanitize.
        Returns:
            str: A sanitized version of the input string suitable for use in filenames.
        """
        return re.sub(r'[^\w\-]', '_', s)

class DecisionTreePlotter:
    """
    Class for plotting decision trees.
    """
    def __init__(
        self,
        clf: DecisionTreeClassifier,
        var: str | list[str],
        reference_var: str | list[str],
        ) -> None:
        """Initialize the TreePlotter with a classifier and configuration."""
        self.clf = clf
        self.cfg = ConfigSingleton.get()
        self.var = var
        self.reference_var = reference_var

    def plot(
        self,
        ) -> None:
        """Plot the decision tree."""
        plt.figure(
            figsize = (
                self.cfg.plotting.figsize.width,
                self.cfg.plotting.figsize.height,
                )
            )
        plot_tree(
            self.clf,
            feature_names = self.var if isinstance(
                self.var,
                list
                ) else [self.var],
            class_names = self.reference_var if isinstance(
                self.reference_var,
                list
                ) else [self.reference_var],
            filled = True,
            rounded = True,
            max_depth = self.cfg.hemorrhage.model.max_depth,
            fontsize = self.cfg.plotting.fontsize.tree,
        )
        plt.title("Decision Tree for Suspect Identification")
        plt.tight_layout()
        if self.cfg.hemorrhage.tree_plot.save:
            os.makedirs(self.cfg.hemorrhage.tree_plot.save_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            filename = f"decision_tree_{timestamp}.png"
            plt.savefig(
                os.path.join(
                    self.cfg.hemorrhage.tree_plot.save_path,
                    filename,
                ),
                dpi = self.cfg.hemorrhage.tree_plot.dpi,
            )
        else:
            print("Tree plot will not be saved as per configuration.")
            plt.show(block = False)