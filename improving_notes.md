# Improving Notes — AFib Risk Calculator

---

## Step 1 — Scorecard Fix for Continuous Variables

**Status:** Done and tested.

**Problem fixed:** `predict_from_scorecard()` previously added a feature's points if the key merely *existed* in the patient dict — ignoring the actual value. A patient with LVEDD=45 and LVEDD=65 got identical scorecard points. Only binary variables behaved correctly.

**Approach chosen:** Option B — `points_per_unit`. Each feature stores a float multiplier; at inference: `score += points_per_unit × actual_value`. Works correctly for both continuous and binary (0/1) variables.

**Files changed:**
- `risk_model.py` — `_compute_point_scores()`: now stores `{points_per_unit: float, type: continuous/binary}` per feature; intercept is scaled by the same factor (`intercept_scaled`)
- `risk_model.py` — `__init__` / `__reset__`: added `intercept_scaled` attribute
- `risk_model.py` — `export_scorecard()`: includes `intercept_scaled` in JSON output
- `risk_model.py` — `predict_risk()`: uses `points_per_unit * patient_value` and `intercept_scaled`
- `predict.py` — `predict_from_scorecard()`: reads `info['points_per_unit']`, multiplies by patient value

**Scorecard formula:** `total_score = intercept_scaled + Σ(points_per_unit_i × x_i)`
where `points_per_unit_i = (β_i / min|β|) × scaling_factor`

**Verified:** High-risk patient score (-2228) vs low-risk (-3099) — correct direction (less negative = higher AFib probability). Two patients with different LVEDD values now produce different scorecard scores. ✓

---

## Step 2 — Stratified Train/Test Split + Index Bug Fix

**Status:** Done and tested.

**Bug fixed (prerequisite):** `remove_outliers_iqr` in `core.py` called `.reset_index(drop=True)` after filtering. This replaced original patient indices (e.g., row 23, 47, 91) with positional integers (0, 1, 2). The index-intersection logic in `_prepare_features()` then matched the wrong patients across features. Fix: removed `.reset_index(drop=True)` — original indices are now preserved.

**Split added:** After `_prepare_features()` returns clean `(x, y)`, a stratified 80/20 split is performed:
- 1884 complete samples → 1507 train + 377 test
- AFib+ rate: 5.9% overall / 5.9% train / 5.8% test — stratification holds
- `self.x_test`, `self.y_test` stored as instance attributes for Step 5 evaluation
- Model is fitted only on `x_train`

**Files changed:**
- `core.py` — `remove_outliers_iqr()`: removed `.reset_index(drop=True)`
- `risk_model.py` — `__init__` / `__reset__`: added `self.x_test`, `self.y_test`
- `risk_model.py` — `train()`: added `train_test_split`, updated `training_data_info` with per-split counts, model now fits on `x_train` only

**Verified:** 4 automated tests — index preservation, split sizes, stratification ratio, model prediction on test set. All PASS. ✓

---

## Step 3 — Stratified 5-Fold Cross-Validation

**Status:** Done and tested.

**What was added:** `cross_validate_model(x, y, n_splits=5)` method on `RiskScoreGenerator`. Runs stratified K-Fold CV on the full dataset (1884 samples) before the train/test split. Reports AUC-ROC, sensitivity, specificity per fold, stored as mean ± std in `self.cv_results` and exported into the scorecard JSON under `cv_performance`.

**Bug fixed:** `make_scorer(roc_auc_score, needs_proba=True)` is not supported in the installed sklearn version. Fixed with `response_method='predict_proba'` instead.

**Files changed:**
- `risk_model.py` — added imports: `StratifiedKFold`, `cross_validate`, `make_scorer`, `roc_auc_score`, `recall_score`
- `risk_model.py` — `__init__` / `__reset__`: added `self.cv_results = {}`
- `risk_model.py` — new method `cross_validate_model()` before `train()`
- `risk_model.py` — `train()`: calls `cross_validate_model(x, y)` after `_prepare_features()`, before the 80/20 split
- `risk_model.py` — `export_scorecard()`: scorecard JSON now includes `cv_performance` key

**CV Results (5-fold stratified, no class balancing):**
| Metric | Mean | Std |
|---|---|---|
| AUC-ROC | 0.6156 | ±0.0570 |
| Sensitivity | 0.0000 | ±0.0000 |
| Specificity | 1.0000 | ±0.0000 |
| Fold AUCs | [0.6196, 0.6565, 0.5044, 0.6470, 0.6504] | |

**Key finding:** Sensitivity = 0 means the model (without class balancing) ignores the AFib-positive minority class entirely at the default threshold 0.5. AUC 0.62 indicates the predicted probabilities have some discriminative power, but the decision boundary is useless. This confirms Step 4 (`class_weight='balanced'`) is urgent.

**Verified:** 12 automated tests — all PASS. ✓

---

## Step 4 — Class Imbalance Handling (`class_weight='balanced'`)

**Status:** Done and tested.

**What was added:** `class_weight='balanced'` to both `LogisticRegression` instances in `risk_model.py`:
- `cross_validate_model()` — CV model
- `train()` — final model

This assigns each class a weight of `n_samples / (n_classes × bincount(y))`. At ~6% AFib+ prevalence, the positive class receives ~15× higher weight. Consistent with the inverse-prevalence weighting already used in `analyze.py`.

**Files changed:**
- `risk_model.py` — `cross_validate_model()`: added `class_weight='balanced'` to `cv_model`
- `risk_model.py` — `train()`: added `class_weight='balanced'` to `self.model`

**Before vs. after (5-fold CV):**
| Metric | Before | After |
|---|---|---|
| AUC-ROC | 0.6156 ± 0.057 | 0.6243 ± 0.055 |
| Sensitivity | 0.000 ± 0.000 | **0.595 ± 0.089** |
| Specificity | 1.000 ± 0.000 | 0.632 ± 0.019 |

**Verified:** Existing 12 CV tests — all PASS. ✓

---

## Step 5 — Performance Metrics Export (held-out test set)

**Status:** Done and tested.

**What was added:** New method `evaluate_on_test_set()` on `RiskScoreGenerator`. Called automatically at end of `train()`. Evaluates the final model on `self.x_test / self.y_test` (the 20% held-out set from Step 2). Results stored in `self.test_performance` and exported to scorecard JSON under `test_performance`.

**Metrics computed:**

| Metric | Value | Notes |
|---|---|---|
| AUC-ROC | **0.6503** | Threshold-independent; consistent with CV estimate 0.6243 ± 0.055 |
| Brier Score | **0.2369** | High — see note below |
| Optimal threshold | **0.4914** | Youden's index: maximises sensitivity + specificity − 1 |
| Sensitivity | **0.7273** | 16/22 AFib+ cases caught |
| Specificity | **0.6197** | 220/355 negatives correctly identified |
| PPV | **0.1060** | Low due to 5.8% prevalence (expected) |
| NPV | **0.9735** | High — strong negative prediction |
| Confusion matrix | TP=16 TN=220 FP=135 FN=6 | |

**Brier score interpretation:** The no-skill Brier for 5.8% prevalence ≈ 0.055. Our score of 0.237 is significantly higher, meaning the raw probability values are poorly calibrated — the model overestimates probabilities (inflated by `class_weight='balanced'` which trains as if prevalence were 50%). The model discriminates (AUC 0.65) but the absolute percentages are unreliable without calibration. This directly motivates Step 9 (Platt scaling).

**Algorithm — Youden optimal threshold:**
```
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
optimal_threshold = thresholds[argmax(tpr - fpr)]
```

**Files changed:**
- `risk_model.py` — imports: added `roc_curve`, `brier_score_loss`, `confusion_matrix`
- `risk_model.py` — `__init__` / `__reset__`: added `self.test_performance = {}`
- `risk_model.py` — new method `evaluate_on_test_set()` between `cross_validate_model()` and `train()`
- `risk_model.py` — `train()`: calls `self.evaluate_on_test_set()` after `_compute_point_scores()`
- `risk_model.py` — `export_scorecard()`: scorecard JSON now includes `test_performance` key

**Verified:** 15/15 tests PASS. ✓

---

## Step 6 — Bootstrap Confidence Intervals

**Status:** Done and tested.

**What was added:** Method `_compute_bootstrap_ci()` on `RiskScoreGenerator`. Called automatically from `evaluate_on_test_set()`. Runs 1000 iterations of stratified bootstrap resampling on the held-out test set, then computes 95% percentile CIs for AUC-ROC, sensitivity, and specificity. Results merged into `self.test_performance` and exported to scorecard JSON.

**Bootstrap design choices:**
- **Stratified:** Resamples within AFib+ and AFib− classes separately (`rng.choice` on `pos_idx` and `neg_idx`), so each bootstrap sample preserves the ~5.8% AFib+ ratio. This prevents degenerate all-negative samples on a small test set.
- **Fixed threshold:** Sensitivity/specificity evaluated at the Youden-optimal threshold from the main evaluation (`self.test_performance['optimal_threshold']`). This gives CIs for performance at a fixed operating point — the correct interpretation for a deployed clinical tool.
- **Seed:** `np.random.default_rng(42)` — reproducible CIs across runs.

**Results (n=1000 stratified bootstrap):**

| Metric | Point estimate | 95% CI |
|---|---|---|
| AUC-ROC | 0.6503 | [0.5474 – 0.7480] |
| Sensitivity | 0.7273 | [0.5455 – 0.9091] |
| Specificity | 0.6197 | [0.5690 – 0.6704] |

**Key observations:**
- AUC 95% CI spans 0.55 – 0.75 — statistically above chance (lower bound > 0.5), but wide due to small test set (22 AFib+)
- Sensitivity CI is wide ([0.55 – 0.91]) — 22 positives is the fundamental limit; each missed/caught case shifts the estimate by ~4.5%
- Specificity CI is narrower ([0.57 – 0.67]) — 355 negatives provide more stable estimates

**Files changed:**
- `risk_model.py` — new method `_compute_bootstrap_ci()` after `evaluate_on_test_set()`
- `risk_model.py` — `evaluate_on_test_set()`: calls `_compute_bootstrap_ci()`, merges CI dict into `self.test_performance`, updated print output to show CIs inline

**Verified:** 15/15 tests PASS. ✓

---

## Step 7 — Cross-Validated Feature Selection (RFECV)

**Status:** Done and tested.

**What was added:** Method `_select_features_rfecv(x, y)` on `RiskScoreGenerator`. Called from `train()` after `_prepare_features()` and before CV and train/test split. Uses `RFECV` (Recursive Feature Elimination with Cross-Validation) to find the optimal feature subset by iteratively removing features and evaluating AUC via 5-fold stratified CV. After selection, `self.model_features` is updated to the selected subset — all downstream methods (CV, split, fit, `_compute_point_scores`, `predict_risk`) automatically operate on the reduced feature set.

**RFECV configuration:**
- Estimator: `LogisticRegression(class_weight='balanced', max_iter=1000)`
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Scoring: `roc_auc`
- `min_features_to_select=3` — floor to prevent over-reduction

**Feature selection result:**

| | Features |
|---|---|
| Original (8) | LVEDD, IVS, CHA2DS2-VASc, LVEF, BMI, Typ ischemie (Teritorialni), Hyperlipidemie, Serovy kreatinin |
| **Selected (3)** | **CHA2DS2-VASc, Typ ischemie (Teritorialni), Hyperlipidemie** |
| Dropped (5) | LVEDD, IVS, LVEF, BMI, Serovy kreatinin |

**Performance after RFECV:**

| Metric | Before Step 7 (8 features) | After Step 7 (3 features) |
|---|---|---|
| AUC-ROC | 0.6503 [0.5474 – 0.7480] | 0.6469 [0.5434 – 0.7398] |

AUC drop is minimal (0.003) — the 5 dropped features contributed negligible discriminative value. Model is now clinically simpler: only 3 inputs required, all clinically interpretable.

**Clinical note:** CHA2DS2-VASc is already a validated stroke/AFib risk score. Its selection as the dominant predictor is clinically coherent. Teritorialni infarkt and Hyperlipidemie are known AFib risk factors.

**EPV impact:** 22 AFib+ cases / 3 features = EPV 7.3 (previously 22/8 = 2.75). Step 8 will evaluate this.

**Files changed:**
- `risk_model.py` — imports: added `RFECV` from `sklearn.feature_selection`
- `risk_model.py` — `__init__` / `__reset__`: added `self.selected_features = []`
- `risk_model.py` — new method `_select_features_rfecv()` before `train()`
- `risk_model.py` — `train()`: calls `_select_features_rfecv()` after `_prepare_features()`, updates `self.model_features`, stores selection info in `training_data_info`

**Verified:** 16/16 tests PASS. ✓

---

## Step 8 — EPV Check (Events Per Variable)

**Status:** Done and tested.

**What was added:** EPV check in `train()` immediately after the train/test split. EPV = `y_train.sum() / len(self.model_features)` (uses training set positives, since that is what the model actually fits on). Results stored in `training_data_info` under `epv` and `epv_warning` (bool). Warning is printed unconditionally if EPV < 10, regardless of `print_progress` setting.

**EPV results:**

| | Before Step 7 (8 features) | After Step 7 (3 features) |
|---|---|---|
| Train AFib+ | 89 | 89 |
| Features | 8 | 3 |
| EPV | 11.1 | **29.7** |
| Warning | No | **No** |

EPV 29.7 is comfortably above the recommended minimum of 10. Note: the 22 AFib+ cases visible in the test set are separate — total positives in full dataset = 89 + 22 = 111 (5.9%). RFECV (Step 7) more than doubled the EPV as a side effect of reducing features.

**Files changed:**
- `risk_model.py` — `train()`: EPV calculation and warning after split, stored in `training_data_info`

**Verified:** 9/9 tests PASS. ✓

---

## Step 9 — Probability Calibration (Platt Scaling)

**Status:** Done and tested.

**What was added:** Method `_calibrate_model()` on `RiskScoreGenerator`. Called at end of `train()` after `evaluate_on_test_set()`. Applies Platt scaling (sigmoid calibration) via `CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')`, fitted on `x_test / y_test`. The base model never saw `x_test` during logistic regression coefficient fitting, so it acts as a calibration set. Stores reliability diagram data and Brier scores pre/post calibration. `predict_risk()` now returns calibrated probabilities.

**Calibration results:**

| | Brier Score |
|---|---|
| Uncalibrated | 0.2394 |
| **Calibrated** | **0.0545** |
| No-skill baseline (~6% prev) | ~0.056 |
| Improvement | 0.1849 |

Calibrated Brier (0.0545) is essentially at the no-skill baseline — meaning the model correctly identifies that most patients have low AFib risk (~6%) and assigns probabilities accordingly. The dramatic drop from 0.239 confirms the `class_weight='balanced'` inflation was entirely corrected.

**Reliability diagram (calibrated, 5 quantile bins):**

| Predicted prob | Actual fraction |
|---|---|
| 0.037 | 0.010 |
| 0.046 | 0.066 |
| 0.054 | 0.051 |
| 0.067 | 0.106 |
| 0.097 | 0.083 |

Probabilities are in a clinically realistic range (3–10%), reflecting true ~6% prevalence. The model is now usable as a calibrated risk score.

**Important limitation:** Brier score 0.0545 is in-sample for the calibration step (fitted on x_test, evaluated on x_test). This is optimistic. Independent validation on new patients would be required before clinical deployment.

**Files changed:**
- `risk_model.py` — imports: added `CalibratedClassifierCV`, `calibration_curve`, `FrozenEstimator`
- `risk_model.py` — `__init__` / `__reset__`: added `self.calibrated_model`, `self.calibration_results`
- `risk_model.py` — new method `_calibrate_model()` before `_compute_point_scores()`
- `risk_model.py` — `train()`: calls `_calibrate_model()` after `evaluate_on_test_set()`
- `risk_model.py` — `predict_risk()`: uses `calibrated_model` if available, else falls back to base model
- `risk_model.py` — `export_scorecard()`: saves calibrated model as `_calibrated.joblib`, adds `calibration` key to scorecard JSON

**Verified:** 18/18 tests PASS. ✓

---
