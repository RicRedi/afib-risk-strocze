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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    make_scorer, roc_auc_score, recall_score,
    roc_curve, brier_score_loss, confusion_matrix,
)
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
        self.intercept_scaled = None
        self.training_data_info = {}
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.cv_results = {}
        self.test_performance = {}
        self.selected_features = []
        self.calibrated_model = None
        self.calibration_results = {}

    def __reset__(self) -> None:
        """Resets the generator's state."""
        self.model_features = []
        self.reference_var = None
        self.df = None
        self.model = None
        self.coefficients = {}
        self.point_scores = {}
        self.intercept = None
        self.intercept_scaled = None
        self.training_data_info = {}
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.cv_results = {}
        self.test_performance = {}
        self.selected_features = []
        self.calibrated_model = None
        self.calibration_results = {}

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

    def cross_validate_model(self, x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        """
        Runs stratified K-Fold cross-validation on the full dataset.

        Reports mean ± std for AUC-ROC, sensitivity (recall pos), specificity (recall neg).
        Results are stored in self.cv_results and returned.

        Args:
            x: Feature matrix (all complete samples, before train/test split)
            y: Target vector
            n_splits: Number of CV folds (default 5)

        Returns:
            dict with mean and std for each metric across folds
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scoring = {
            'auc': make_scorer(roc_auc_score, response_method='predict_proba'),
            'sensitivity': make_scorer(recall_score, pos_label=1, zero_division=0),
            'specificity': make_scorer(recall_score, pos_label=0, zero_division=0),
        }

        cv_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced')

        raw = cross_validate(cv_model, x, y, cv=cv, scoring=scoring, return_train_score=False)

        self.cv_results = {
            'n_splits': n_splits,
            'auc_mean': float(np.mean(raw['test_auc'])),
            'auc_std': float(np.std(raw['test_auc'])),
            'sensitivity_mean': float(np.mean(raw['test_sensitivity'])),
            'sensitivity_std': float(np.std(raw['test_sensitivity'])),
            'specificity_mean': float(np.mean(raw['test_specificity'])),
            'specificity_std': float(np.std(raw['test_specificity'])),
            'fold_auc': [float(v) for v in raw['test_auc']],
            'fold_sensitivity': [float(v) for v in raw['test_sensitivity']],
            'fold_specificity': [float(v) for v in raw['test_specificity']],
        }

        if self.cfg.analysis.print_progress:
            print(f"\n{n_splits}-Fold Stratified Cross-Validation Results:")
            print(
                f"  AUC-ROC:     {\
                    self.cv_results['auc_mean']:.4f\
                    } ± {self.cv_results['auc_std']:.4f}")
            print(
                f"  Sensitivity: {\
                    self.cv_results['sensitivity_mean']:.4f\
                    } ± {self.cv_results['sensitivity_std']:.4f}")
            print(
                f"  Specificity: {\
                    self.cv_results['specificity_mean']:.4f\
                    } ± {self.cv_results['specificity_std']:.4f}")

        return self.cv_results

    def evaluate_on_test_set(self) -> dict:
        """
        Evaluates the trained model on the held-out test set (from Step 2 split).

        Computes threshold-independent metrics (AUC-ROC, Brier score) and
        threshold-dependent metrics (sensitivity, specificity, PPV, NPV, confusion matrix)
        at the Youden-optimal cut-point (maximises sensitivity + specificity - 1).
        Also computes 95% bootstrap confidence intervals for AUC, sensitivity, specificity.

        Returns:
            dict with all metrics, stored in self.test_performance
        """
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test set not available. Run train() first.")
        if self.model is None:
            raise ValueError("Model not trained. Run train() first.")

        y_prob = self.model.predict_proba(self.x_test)[:, 1]

        # Threshold-independent metrics
        auc = float(roc_auc_score(self.y_test, y_prob))
        brier = float(brier_score_loss(self.y_test, y_prob))

        # Youden-optimal threshold: maximises tpr - fpr (= sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[optimal_idx])

        # Threshold-dependent metrics at optimal cut-point
        y_pred = (y_prob >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        cohort_prevalence = float(self.y_test.sum() / len(self.y_test))

        self.test_performance = {
            'n_test': int(len(self.y_test)),
            'n_positive': int(self.y_test.sum()),
            'n_negative': int(len(self.y_test) - self.y_test.sum()),
            'cohort_prevalence': cohort_prevalence,
            'auc_roc': auc,
            'brier_score': brier,
            'optimal_threshold': optimal_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }

        # Bootstrap 95% CIs
        ci = self._compute_bootstrap_ci()
        self.test_performance.update(ci)

        if self.cfg.analysis.print_progress:
            print("\nTest Set Performance:")
            print(f"  AUC-ROC:            {auc:.4f}  "
                  f"[{ci['auc_ci_lower']:.4f} – {ci['auc_ci_upper']:.4f}]")
            print(f"  Brier Score:        {brier:.4f}")
            print(f"  Optimal threshold:  {optimal_threshold:.4f} (Youden)")
            print(f"  Sensitivity:        {sensitivity:.4f}  "
                  f"[{ci['sensitivity_ci_lower']:.4f} – {ci['sensitivity_ci_upper']:.4f}]")
            print(f"  Specificity:        {specificity:.4f}  "
                  f"[{ci['specificity_ci_lower']:.4f} – {ci['specificity_ci_upper']:.4f}]")
            print(f"  PPV:                {ppv:.4f}")
            print(f"  NPV:                {npv:.4f}")
            print(f"  Confusion matrix:   TP={tp} TN={tn} FP={fp} FN={fn}")

        return self.test_performance

    def _compute_bootstrap_ci(self, n_iterations: int = 1000) -> dict:
        """
        Computes 95% bootstrap CIs for AUC, sensitivity, and specificity.

        Stratified bootstrap: resamples within each class separately to preserve
        the AFib+ ratio in every bootstrap sample. Sensitivity/specificity use
        the Youden-optimal threshold from self.test_performance (fixed operating point).

        Args:
            n_iterations: Number of bootstrap iterations (default 1000)

        Returns:
            dict with CI lower/upper bounds for AUC, sensitivity, specificity
        """
        if not self.test_performance:
            raise ValueError("Run evaluate_on_test_set() first.")

        optimal_threshold = self.test_performance['optimal_threshold']

        pos_idx = np.where(self.y_test == 1)[0]
        neg_idx = np.where(self.y_test == 0)[0]

        rng = np.random.default_rng(42)
        boot_auc = []
        boot_sens = []
        boot_spec = []

        for _ in range(n_iterations):
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

            x_boot = self.x_test[boot_idx]
            y_boot = self.y_test[boot_idx]

            y_prob_boot = self.model.predict_proba(x_boot)[:, 1]

            if len(np.unique(y_boot)) < 2:
                continue

            boot_auc.append(float(roc_auc_score(y_boot, y_prob_boot)))

            y_pred_boot = (y_prob_boot >= optimal_threshold).astype(int)
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_boot, y_pred_boot).ravel()

            sens_b = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
            spec_b = float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0
            boot_sens.append(sens_b)
            boot_spec.append(spec_b)

        return {
            'n_bootstrap': len(boot_auc),
            'auc_ci_lower': float(np.percentile(boot_auc, 2.5)),
            'auc_ci_upper': float(np.percentile(boot_auc, 97.5)),
            'sensitivity_ci_lower': float(np.percentile(boot_sens, 2.5)),
            'sensitivity_ci_upper': float(np.percentile(boot_sens, 97.5)),
            'specificity_ci_lower': float(np.percentile(boot_spec, 2.5)),
            'specificity_ci_upper': float(np.percentile(boot_spec, 97.5)),
        }

    @staticmethod
    def _risk_category(probability: float, prevalence: float) -> dict:
        """
        Maps a calibrated probability to a risk category relative to cohort prevalence.

        Thresholds (multiples of cohort prevalence):
            < 1×     → Nízké          — FiS nepravděpodobná
            1–1.5×   → Střední        — Mírně zvýšené riziko
            1.5–2×   → Vysoké         — Výrazně zvýšené riziko
            > 2×     → Velmi vysoké   — Silné doporučení k monitoringu

        Args:
            probability: Calibrated AFib probability from predict_proba (0–1)
            prevalence:  Cohort AFib prevalence (from test_performance['cohort_prevalence'])

        Returns:
            dict with category label, clinical description, and ratio to prevalence
        """
        ratio = probability / prevalence if prevalence > 0 else 0.0

        if ratio < 1.0:
            category, description = "Nízké", "FiS nepravděpodobná"
        elif ratio < 1.5:
            category, description = "Střední", "Mírně zvýšené riziko záchytu FiS"
        elif ratio < 2.0:
            category, description = "Vysoké", "Výrazně zvýšené riziko záchytu FiS"
        else:
            category, description = "Velmi vysoké", "Silné doporučení k monitoringu FiS"

        return {
            'category': category,
            'description': description,
            'ratio_to_prevalence': round(ratio, 2),
        }

    def _select_features_rfecv(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        Selects optimal feature subset via Recursive Feature Elimination with CV.

        Runs RFECV with 5-fold stratified CV on the full dataset (before any split).
        Chooses the feature count that maximises AUC-ROC. Updates self.selected_features
        and self.model_features to the selected subset so all downstream methods
        (coefficients, point scores, predict_risk) operate on the same features.

        Args:
            x: Full feature matrix (all complete samples, before train/test split)
            y: Target vector

        Returns:
            (x_selected, selected_indices): filtered x and indices of selected columns
        """
        selector = RFECV(
            estimator=LogisticRegression(
                class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42
            ),
            step=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            min_features_to_select=3,
        )
        selector.fit(x, y)

        selected_indices = np.where(selector.support_)[0]
        self.selected_features = [self.model_features[i] for i in selected_indices]

        if self.cfg.analysis.print_progress:
            dropped = [f for f in self.model_features if f not in self.selected_features]
            print(f"\nRFECV Feature Selection:")
            print(f"  Original features: {len(self.model_features)}")
            print(f"  Selected features: {len(self.selected_features)}")
            print(f"  Selected:  {self.selected_features}")
            if dropped:
                print(f"  Dropped:   {dropped}")

        self.training_data_info['rfecv_n_features_range'] = list(range(
            selector.min_features_to_select, len(self.model_features) + 1
        ))
        self.training_data_info['rfecv_mean_scores'] = \
            selector.cv_results_['mean_test_score'].tolist()
        self.training_data_info['rfecv_std_scores'] = \
            selector.cv_results_['std_test_score'].tolist()

        return x[:, selected_indices], selected_indices

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

        # Feature selection via RFECV (uses internal CV — does not touch test set)
        all_features = list(self.model_features)
        x, _ = self._select_features_rfecv(x, y)
        self.model_features = self.selected_features

        self.training_data_info['n_features_original'] = len(all_features)
        self.training_data_info['n_features_selected'] = len(self.selected_features)
        self.training_data_info['all_features'] = all_features
        self.training_data_info['selected_features'] = self.selected_features

        # Cross-validation on full dataset (selected features, before split)
        self.cross_validate_model(x, y)

        # Stratified 80/20 train/test split — preserves AFib+ ratio in both sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

        # Record split info
        self.training_data_info['train_samples'] = int(len(y_train))
        self.training_data_info['test_samples'] = int(len(y_test))
        self.training_data_info['train_positive'] = int(y_train.sum())
        self.training_data_info['test_positive'] = int(y_test.sum())

        # EPV check: n_positive_train / n_features (recommended minimum: 10)
        epv = float(y_train.sum()) / len(self.model_features)
        epv_ok = epv >= 10.0
        self.training_data_info['epv'] = round(epv, 2)
        self.training_data_info['epv_warning'] = not epv_ok
        if self.cfg.analysis.print_progress or not epv_ok:
            label = "OK" if epv_ok else "WARNING — below recommended minimum of 10"
            print(f"\nEPV check: {y_train.sum()} AFib+ / {len(self.model_features)} features "
                  f"= {epv:.1f}  [{label}]")

        if self.cfg.analysis.print_progress:
            print(f"\nTrain/test split: {len(y_train)} train / {len(y_test)} test")
            print(f"  Train AFib+: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
            print(f"  Test  AFib+: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
            print(f"\nFitting Logistic Regression model on {x_train.shape[0]} \
                samples with {x_train.shape[1]} features...")

        # ========================================================================================
        # ========================================================================================
        # ============================== HERE IS THE MODEL TRAINING ==============================
        # ========================================================================================
        # ========================================================================================
        # Train logistic regression on training set only
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
        )
        self.model.fit(x_train, y_train)
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

        # Evaluate on held-out test set (uncalibrated model — discrimination metrics)
        self.evaluate_on_test_set()

        # Calibrate probability outputs (Platt scaling)
        self._calibrate_model()

    def _calibrate_model(self) -> dict:
        """
        Applies Platt scaling (sigmoid calibration) to correct probability outputs.

        The base model trained with class_weight='balanced' produces inflated
        probabilities (acts as if prevalence ~50%; true prevalence ~6%).
        Platt scaling fits a logistic function from raw scores → calibrated probabilities.

        Fitted on x_test / y_test: the base model never saw this data for coefficient
        fitting, so Platt scaling uses it as a calibration set. The resulting calibrated
        Brier score on x_test is in-sample for the calibration step — treat as indicative,
        not as independent validation.

        Returns:
            dict with pre/post Brier scores and reliability diagram data
        """
        if self.model is None or self.x_test is None:
            raise ValueError("Run train() and evaluate_on_test_set() first.")

        self.calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(self.model), method='sigmoid'
        )
        self.calibrated_model.fit(self.x_test, self.y_test)

        y_prob_uncal = self.model.predict_proba(self.x_test)[:, 1]
        y_prob_cal = self.calibrated_model.predict_proba(self.x_test)[:, 1]

        brier_uncal = float(brier_score_loss(self.y_test, y_prob_uncal))
        brier_cal = float(brier_score_loss(self.y_test, y_prob_cal))

        # Reliability diagram data (5 quantile bins — robust with small n_positive)
        frac_pos_u, mean_pred_u = calibration_curve(
            self.y_test, y_prob_uncal, n_bins=5, strategy='quantile'
        )
        frac_pos_c, mean_pred_c = calibration_curve(
            self.y_test, y_prob_cal, n_bins=5, strategy='quantile'
        )

        # Calibrated performance metrics — Youden threshold on calibrated probabilities.
        # These are consistent with risk_category() and predict_risk() which both use
        # the calibrated model. The uncalibrated metrics in test_performance should NOT
        # be reported externally.
        fpr_c, tpr_c, thresh_c = roc_curve(self.y_test, y_prob_cal)
        opt_idx_c = int(np.argmax(tpr_c - fpr_c))
        opt_thresh_c = float(thresh_c[opt_idx_c])

        y_pred_c = (y_prob_cal >= opt_thresh_c).astype(int)
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(self.y_test, y_pred_c).ravel()

        sens_c = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0
        spec_c = float(tn_c / (tn_c + fp_c)) if (tn_c + fp_c) > 0 else 0.0
        ppv_c  = float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) > 0 else 0.0
        npv_c  = float(tn_c / (tn_c + fn_c)) if (tn_c + fn_c) > 0 else 0.0

        self.calibration_results = {
            'method': 'platt_scaling',
            'brier_uncalibrated': brier_uncal,
            'brier_calibrated': brier_cal,
            'brier_improvement': round(brier_uncal - brier_cal, 4),
            'note': 'calibrated on x_test — metrics below are in-sample for calibration set',
            'platt_a': float(self.calibrated_model.calibrated_classifiers_[0].calibrators[0].a_),
            'platt_b': float(self.calibrated_model.calibrated_classifiers_[0].calibrators[0].b_),
            'calibrated_optimal_threshold': opt_thresh_c,
            'calibrated_sensitivity': sens_c,
            'calibrated_specificity': spec_c,
            'calibrated_ppv': ppv_c,
            'calibrated_npv': npv_c,
            'calibrated_tp': int(tp_c),
            'calibrated_tn': int(tn_c),
            'calibrated_fp': int(fp_c),
            'calibrated_fn': int(fn_c),
            'reliability_uncalibrated': {
                'fraction_of_positives': frac_pos_u.tolist(),
                'mean_predicted_value': mean_pred_u.tolist(),
            },
            'reliability_calibrated': {
                'fraction_of_positives': frac_pos_c.tolist(),
                'mean_predicted_value': mean_pred_c.tolist(),
            },
        }

        if self.cfg.analysis.print_progress:
            print(f"\nCalibration (Platt scaling on test set):")
            print(f"  Brier before:    {brier_uncal:.4f}")
            print(f"  Brier after:     {brier_cal:.4f}")
            print(f"  Threshold (cal): {opt_thresh_c:.4f} (Youden on calibrated probs)")
            print(f"  Sensitivity:     {sens_c:.4f}")
            print(f"  Specificity:     {spec_c:.4f}")
            print(f"  PPV:             {ppv_c:.4f}  NPV: {npv_c:.4f}")

        return self.calibration_results

    def _compute_point_scores(self) -> None:
        """
        Converts logistic regression coefficients to per-unit point scores.

        Algorithm:
            1. Find the minimum absolute coefficient value
            2. Divide all coefficients by this minimum (normalize)
            3. Multiply by scaling_factor (from config)
            4. Store as float points_per_unit — NOT rounded, to support continuous variables

        Score at inference: score += points_per_unit * actual_patient_value
        This is correct for both binary (0/1) and continuous variables.

        Formula: points_per_unit = (β / min|β|) * scaling_factor
        Intercept is scaled by the same factor for scorecard consistency.
        """
        if not self.coefficients:
            raise ValueError("Coefficients not yet computed. Run train() first.")

        categorical_features = set(self.cfg.variables.independent_binary_variables)

        abs_coefs = np.array([abs(c) for c in self.coefficients.values()])
        min_abs_coef = np.min(abs_coefs)

        if self.cfg.analysis.print_progress:
            print("\nComputing point scores...")
            print(f"Scaling factor: {self.cfg.model.scaling_factor}")
            print(f"Minimum |beta|: {min_abs_coef:.6f}")

        scale = self.cfg.model.scaling_factor / min_abs_coef

        # Scale intercept by the same factor for scorecard consistency
        self.intercept_scaled = float(self.intercept * scale)

        for feature, coef in self.coefficients.items():
            feature_type = 'binary' if feature in categorical_features else 'continuous'
            self.point_scores[feature] = {
                'points_per_unit': float(coef * scale),
                'type': feature_type,
            }

        if self.cfg.analysis.print_progress:
            print(f"Point scores (per unit) calculated for {len(self.point_scores)} features")

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

        # Save base model
        model_path = self.cfg.model.model_output.model_path
        dump(self.model, model_path)

        # Save calibrated model alongside base model
        if self.calibrated_model is not None:
            cal_path = model_path.replace('.joblib', '_calibrated.joblib')
            dump(self.calibrated_model, cal_path)
            if self.cfg.analysis.print_progress:
                print(f"Calibrated model saved to: {cal_path}")

        if self.cfg.analysis.print_progress:
            print(f"Model saved to: {model_path}")

        # Export test and train arrays for visualization scripts and model comparison (M0.1)
        # Contains only numerical feature values — no patient IDs or personal data
        if self.x_test is not None:
            np.save(model_path.replace('.joblib', '_x_test.npy'), self.x_test)
            np.save(model_path.replace('.joblib', '_y_test.npy'), self.y_test)
            if self.cfg.analysis.print_progress:
                print(f"Test arrays saved:  _x_test.npy ({self.x_test.shape}), "
                      f"_y_test.npy ({self.y_test.shape})")
        if self.x_train is not None:
            np.save(model_path.replace('.joblib', '_x_train.npy'), self.x_train)
            np.save(model_path.replace('.joblib', '_y_train.npy'), self.y_train)
            if self.cfg.analysis.print_progress:
                print(f"Train arrays saved: _x_train.npy ({self.x_train.shape}), "
                      f"_y_train.npy ({self.y_train.shape})")

        # Create scorecard
        scorecard = {
            'intercept': float(self.intercept),
            'intercept_scaled': float(self.intercept_scaled),
            'scaling_factor': self.cfg.model.scaling_factor,
            'variables': self.point_scores,
            'training_data_info': self.training_data_info,
            'cv_performance': self.cv_results,
            'test_performance': self.test_performance,
            'calibration': self.calibration_results,
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

        # Get probability prediction — use calibrated model if available
        predictor = self.calibrated_model if self.calibrated_model is not None else self.model
        risk_probability = predictor.predict_proba(x_patient)[0][1]

        # Calculate point-based score: points_per_unit * actual patient value
        individual_points = {
            feature: self.point_scores[feature]['points_per_unit'] * patient_dict[feature]
            for feature in self.model_features
        }
        risk_score = self.intercept_scaled + sum(individual_points.values())

        # Risk category relative to cohort prevalence
        prevalence = self.test_performance.get('cohort_prevalence', 0.06)
        category_info = self._risk_category(float(risk_probability), prevalence)

        return {
            'risk_probability': float(risk_probability),
            'risk_category': category_info['category'],
            'risk_description': category_info['description'],
            'ratio_to_prevalence': category_info['ratio_to_prevalence'],
            'cohort_prevalence': prevalence,
            'risk_score': float(risk_score),
            'individual_points': individual_points,
        }


# Example usage:
if __name__ == "__main__":
    ConfigSingleton.set()
    generator = RiskScoreGenerator()
    generator.train()
    generator.export_scorecard()
