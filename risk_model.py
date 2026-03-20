# -*- coding: utf-8 -*-

"""
Created on 19. 03. 2026 at 00:00:00

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
    Module for training a clinical risk score model using logistic regression.
    Fits a model on fixed variables, converts coefficients to clinical point scores,
    and exports results for deployment.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump#, load

from core import (
    validate_inputs_from_signature,
    convert_column_to_binary,
    remove_outliers_iqr,
    load_data,
    evaluate_logic,
)
from utils.config_singleton import ConfigSingleton


class RiskScoreGenerator:
    """
    Class for training a clinical risk score model using logistic regression.
    
    Takes a fixed set of variables, trains a logistic regression model,
    and converts coefficients into a clinical point-based scoring system.
    
    Attributes:
        cfg: Configuration object from ConfigSingleton
        model_features (list): Fixed list of variables for model training
        reference_var (str): Target outcome variable
        df (pd.DataFrame): DataFrame containing the loaded data
        model (LogisticRegression): Fitted logistic regression model
        coefficients (dict): Dictionary mapping features to their coefficients
        point_scores (dict): Dictionary mapping features to integer point values
        intercept (float): Model intercept
    """

    def __init__(self) -> None:
        """Initializes the RiskScoreGenerator with configuration settings."""
        self.cfg = ConfigSingleton.get()
        self.model_features = []
        self.reference_var = None
        self.df = None
        self.model = None
        self.coefficients = {}
        self.point_scores = {}
        self.intercept = None
        self.training_data_info = {}

    def __reset__(self) -> None:
        """Resets the generator's state."""
        self.model_features = []
        self.reference_var = None
        self.df = None
        self.model = None
        self.coefficients = {}
        self.point_scores = {}
        self.intercept = None
        self.training_data_info = {}

    def __repr__(self) -> str:
        """String representation of the RiskScoreGenerator."""
        return (
            f"RiskScoreGenerator("
            f"file_path = {self.cfg.analysis.file_path}, "
            f"model_features = {self.model_features}, "
            f"reference_var = {self.reference_var}, "
            f"scaling_factor = {self.cfg.model.scaling_factor}"
            ")"
        )

    def __load_attr__(self) -> None:
        """
        Loads configuration and data.
        
        Raises:
            KeyError: If required config keys are missing
            ValueError: If model features or reference variable not found
        """
        # Get model features and reference from config
        self.model_features = self.cfg.model.model_features
        self.reference_var = self.cfg.model.reference_var

        if not self.model_features or not self.reference_var:
            raise ValueError(
                "Configuration must contain 'model_features' and 'reference_var'."
            )

        if self.cfg.analysis.print_progress:
            print(f"Loading {len(self.model_features)} model features")
            print("Reference variable configured")

        # Load data
        self.df = load_data(
            self.cfg.analysis.file_path,
            self.model_features,
            self.reference_var,
        )

        if self.cfg.analysis.print_progress:
            print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def _prepare_features(self) -> tuple:
        """
        Prepares feature matrix (X) and target vector (y) for model training.
        
        Handles missing data, outliers, and converts categorical variables to binary.
        Only uses rows where ALL features and target have valid data.
        
        Returns:
            tuple: (X, y, data_completeness_info)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Run __load_attr__() first.")

        # Build list of categorical (binary) features for identification
        categorical_features = set(self.cfg.variables.independent_binary_variables)

        # Prepare target variable (y)
        y = evaluate_logic(
            self.df,
            self.cfg.variables.conditions,
            self.cfg.variables.logic,
        ).astype(int)

        if self.cfg.analysis.print_progress:
            print(f"Target variable created: {y.sum()} positive cases out of {len(y)}")

        # Prepare feature matrix (X)
        x_dict = {}
        data_completeness = {}

        for i, feature in enumerate(self.model_features):
            if self.cfg.analysis.print_progress:
                print(f"Processing feature {i+1}/{len(self.model_features)}")

            # Get feature data
            x = self.df[feature].copy()
            original_count = len(x)

            # Distinguish between continuous and categorical features
            if feature in categorical_features:
                # CATEGORICAL: Handle "Nezjištěno" (unknown) and convert to binary
                x = x.apply(
                    lambda val: np.nan if val == 'Nezjištěno' else val
                ).dropna()

                if len(x) > 0:
                    x = convert_column_to_binary(x, numpy=False)
                    x = x.values  # Convert back to numpy array
            else:
                # CONTINUOUS: Handle strings, infinities, outliers
                x = x.apply(
                    lambda val: np.nan if isinstance(val, str) else val
                ).replace(
                    [np.inf, -np.inf],
                    np.nan
                ).dropna()

                # Remove outliers using IQR
                if len(x) > 0:
                    x = remove_outliers_iqr(
                        x,
                        threshold=self.cfg.analysis.iqr_threshold,
                    )

            # Record data completeness
            data_completeness[feature] = {
                'original_count': original_count,
                'after_cleaning': len(x),
                'completeness_%': np.round(len(x) / original_count * 100, 2)
            }

            x_dict[feature] = x

        # Find common valid indices across all features and target
        valid_indices = None
        for feature in self.model_features:
            feature_indices = x_dict[feature].index if hasattr(x_dict[feature], 'index') \
                else pd.Series(x_dict[feature]).index
            if valid_indices is None:
                valid_indices = set(feature_indices)
            else:
                valid_indices = valid_indices.intersection(feature_indices)

        # Also must have valid target
        valid_y_indices = y[y.notna()].index
        valid_indices = valid_indices.intersection(valid_y_indices)

        if self.cfg.analysis.print_progress:
            print(
                f"Valid samples (all features + target): {len(valid_indices)} out of {len(self.df)}"
                )

        # Build final X and y with common indices
        x_final = []
        for feature in self.model_features:
            feature_data = x_dict[feature]
            if hasattr(feature_data, 'loc'):
                feature_values = feature_data.loc[list(valid_indices)].values
            else:
                feature_values = feature_data[list(valid_indices)]
            x_final.append(feature_values)

        x = np.column_stack(x_final)
        y_final = y.loc[list(valid_indices)].values

        # Store training info
        self.training_data_info = {
            'total_rows_in_dataset': len(self.df),
            'rows_with_complete_data': len(valid_indices),
            'data_completeness_%': np.round(len(valid_indices) / len(self.df) * 100, 2),
            'feature_completeness': data_completeness,
            'target_positive_cases': int(y_final.sum()),
            'target_negative_cases': int(len(y_final) - y_final.sum()),
        }

        if self.cfg.analysis.print_progress:
            print("\nTraining Data Summary:")
            print(f"  Complete samples: {len(valid_indices)} out of {len(self.df)} \
                ({self.training_data_info['data_completeness_%']:.1f}%)")
            print(f"  Target: {self.training_data_info['target_positive_cases']} positive, \
                {self.training_data_info['target_negative_cases']} negative")

        return x, y_final, self.training_data_info

    def train(self) -> None:
        """
        Trains the logistic regression model on prepared features.
        
        Steps:
            1. Load and prepare data
            2. Fit LogisticRegression model
            3. Extract coefficients
            4. Calculate point scores
        """
        if self.cfg.analysis.print_progress:
            print("\n" + "="*60)
            print("Starting Model Training")
            print("="*60)

        validate_inputs_from_signature(self.train, {'self': self})

        self.__reset__()
        self.__load_attr__()

        # Prepare features and target
        x, y, _ = self._prepare_features()

        if len(np.unique(y)) != 2:
            raise ValueError(
                f"Target variable must be binary. Found {len(np.unique(y))} classes."
            )

        if self.cfg.analysis.print_progress:
            print(f"\nFitting Logistic Regression model on {x.shape[0]} \
                samples with {x.shape[1]} features...")

        # ========================================================================================
        # ========================================================================================
        # ============================== HERE IS THE MODEL TRAINING ==============================
        # ========================================================================================
        # ========================================================================================
        # Train logistic regression
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        self.model.fit(x, y)
        # ========================================================================================
        # ========================================================================================
        # ============================== HERE IS THE MODEL TRAINING ==============================
        # ========================================================================================
        # ========================================================================================

        # Extract coefficients
        self.intercept = self.model.intercept_[0]
        self.coefficients = {
            feature: coef
            for feature, coef in zip(self.model_features, self.model.coef_[0])
        }

        if self.cfg.analysis.print_progress:
            print("Model trained successfully!")
            print(f"Intercept: {self.intercept:.4f}")
            print(f"Coefficients extracted for {len(self.coefficients)} features")

        # Calculate point scores
        self._compute_point_scores()

    def _compute_point_scores(self) -> None:
        """
        Converts logistic regression coefficients to integer point scores.
        
        Algorithm:
            1. Find the minimum absolute coefficient value
            2. Divide all coefficients by this minimum
            3. Multiply by scaling_factor (from config)
            4. Round to nearest integer
        
        Formula: points = round((β / min|β|) * scaling_factor)
        """
        if not self.coefficients:
            raise ValueError("Coefficients not yet computed. Run train() first.")

        # Find minimum absolute coefficient
        abs_coefs = np.array([abs(c) for c in self.coefficients.values()])
        min_abs_coef = np.min(abs_coefs)

        if self.cfg.analysis.print_progress:
            print("\nComputing point scores...")
            print(f"Scaling factor: {self.cfg.model.scaling_factor}")
            print(f"Minimum |beta|: {min_abs_coef:.6f}")

        # Normalize and scale
        for feature, coef in self.coefficients.items():
            normalized = coef / min_abs_coef
            scaled = normalized * self.cfg.model.scaling_factor
            points = int(np.round(scaled))
            self.point_scores[feature] = points

        if self.cfg.analysis.print_progress:
            print(f"Point scores calculated for {len(self.point_scores)} features")

    def export_scorecard(self) -> None:
        """
        Exports the trained model and point scorecard.
        
        Outputs:
            1. Model file (.joblib) - serialized sklearn LogisticRegression
            2. Scorecard file (.json) - point table with intercept and feature points
        
        Raises:
            ValueError: If model not yet trained
        """
        if self.model is None or not self.point_scores:
            raise ValueError("Model not yet trained. Run train() first.")

        if self.cfg.analysis.print_progress:
            print("\n" + "="*60)
            print("Exporting Model and Scorecard")
            print("="*60)

        # Save model
        model_path = self.cfg.model.model_output.model_path
        dump(self.model, model_path)

        if self.cfg.analysis.print_progress:
            print(f"Model saved to: {model_path}")

        # Create scorecard
        scorecard = {
            'intercept': float(self.intercept),
            'scaling_factor': self.cfg.model.scaling_factor,
            'variables': self.point_scores,
            'training_data_info': self.training_data_info,
        }

        # Save scorecard
        scorecard_path = self.cfg.model.model_output.scorecard_path
        with open(scorecard_path, 'w', encoding='utf-8') as f:
            json.dump(scorecard, f, ensure_ascii=False, indent=4)

        if self.cfg.analysis.print_progress:
            print(f"Scorecard saved to: {scorecard_path}")
            print("\nScorecard Preview:")
            print(json.dumps(scorecard, ensure_ascii=False, indent=2))

    def predict_risk(self, patient_dict: dict) -> dict:
        """
        Predicts AFib risk for a new patient using the point-based system.
        
        Args:
            patient_dict (dict): Dictionary with feature names as keys and values as data.
                                Example: {'LVEDD': 45.5, 'IVS': 12.0, ...}
        
        Returns:
            dict: Results containing:
                - 'risk_probability': Probability of positive outcome (0-1)
                - 'risk_score': Total points from scorecard
                - 'individual_points': Points for each feature
        
        Raises:
            ValueError: If model not trained or missing features
        """
        if self.model is None:
            raise ValueError("Model not yet trained. Run train() first.")

        # Validate all required features are present
        missing_features = set(self.model_features) - set(patient_dict.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Prepare patient data in correct order
        x_patient = np.array([
            patient_dict[feature] for feature in self.model_features
        ]).reshape(1, -1)

        # Get probability prediction
        risk_probability = self.model.predict_proba(x_patient)[0][1]

        # Calculate point-based score
        individual_points = {
            feature: self.point_scores[feature]
            for feature in self.model_features
        }
        risk_score = self.intercept + sum(individual_points.values())

        return {
            'risk_probability': float(risk_probability),
            'risk_score': float(risk_score),
            'individual_points': individual_points,
        }


# Example usage:
if __name__ == "__main__":
    ConfigSingleton.set()
    generator = RiskScoreGenerator()
    generator.train()
    generator.export_scorecard()
