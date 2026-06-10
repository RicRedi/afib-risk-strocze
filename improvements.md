# Model Improvement Plan — AFib Risk Calculator

**Goal:** A clinically deployable risk calculator — defensible performance, correct scorecard, reproducible pipeline.  
**Constraint:** Single dataset (STROCZE). All validation is internal.

---

## 1. Fix the Scorecard for Continuous Variables

**Problem:** `predict_from_scorecard()` in `predict.py` adds a feature's points if the key merely *exists* in the patient dict, regardless of its value. A patient with LVEDD=45 and one with LVEDD=65 get identical scorecard points. Only binary variables behave correctly.

**Fix:** The scorecard must encode thresholds or a per-unit scaling for continuous variables. Two options:

- **Option A (simpler):** Convert all continuous features to binary via clinically meaningful cut-points (e.g., LVEDD > 55 mm = 1, else 0) before training. The scorecard then works correctly for all features. Cut-points should come from ROC-optimal thresholds or established clinical values, not arbitrary splits.
- **Option B (correct continuous scoring):** Store `points_per_unit` in the scorecard (i.e., the scaled β coefficient) and multiply by the actual patient value at inference: `score += points_per_unit * patient_value`. Requires updating both `_compute_point_scores()` and `predict_from_scorecard()`.

Option A is more defensible clinically (matches how existing scores like CHA₂DS₂-VASc work). Option B is more statistically precise.

---

## 2. Add Stratified Train/Test Split

**Problem:** `risk_model.py` fits on 100% of the data. There is no held-out set, so reported performance (if added) would be optimistic.

**Fix:** Before any model fitting, split data into train (80%) and test (20%) using stratified sampling to preserve the AFib-positive ratio.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # critical: preserves AFib+ ratio in both splits
)
```

The model is trained on `x_train` only. `x_test` / `y_test` are never touched until final evaluation.

---

## 3. Replace Point Estimate with Stratified K-Fold Cross-Validation

**Problem:** With a single train/test split on a small clinical dataset, performance metrics have high variance — the result depends heavily on which patients ended up in the test set.

**Fix:** Use stratified 5-fold or 10-fold cross-validation to estimate performance. Report mean ± std across folds.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model, x, y,
    cv=cv,
    scoring={
        'auc': make_scorer(roc_auc_score, needs_proba=True),
        'sensitivity': make_scorer(recall_score, pos_label=1),
        'specificity': make_scorer(recall_score, pos_label=0),
    },
    return_train_score=False,
)
```

This is the minimum required to quote performance metrics to clinicians.

---

## 4. Add Class Imbalance Handling

**Problem:** `analyze.py` explicitly uses inverse-prevalence weights to handle AFib class imbalance. `risk_model.py` does not — the model will be biased toward predicting the majority class (AFib-negative).

**Fix:** Add `class_weight='balanced'` to `LogisticRegression`. This is equivalent to the inverse-prevalence weighting already used in `analyze.py` and keeps the two parts of the pipeline consistent.

```python
self.model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'   # add this
)
```

---

## 5. Add Performance Metrics Export

**Problem:** `export_scorecard()` saves coefficients and point scores but no model performance. Without metrics, the calculator cannot be defended in a clinical setting.

**Required metrics to compute and export:**

| Metric | Why it matters clinically |
|---|---|
| AUC-ROC | Overall discriminative ability |
| Sensitivity | Proportion of true AFib cases caught |
| Specificity | Proportion of true negatives correctly identified |
| PPV / NPV | What a positive/negative score actually means for this patient |
| Brier Score | Calibration — are predicted probabilities accurate? |
| Optimal threshold | The cut-point that maximises sensitivity+specificity (Youden's index) |

These should be computed on the held-out test set (step 2) and reported per cross-validation fold (step 3), then saved alongside the scorecard JSON.

---

## 6. Add Bootstrap Confidence Intervals for Metrics

**Problem:** Clinical datasets are typically small. A single AUC value without a confidence interval is not credible — reviewers and clinicians will ask for it.

**Fix:** Use bootstrap resampling (1000 iterations) to compute 95% CIs for AUC, sensitivity, and specificity.

```python
from sklearn.utils import resample

auc_scores = []
for _ in range(1000):
    x_boot, y_boot = resample(x_test, y_test, stratify=y_test, random_state=None)
    auc_scores.append(roc_auc_score(y_boot, model.predict_proba(x_boot)[:, 1]))

ci_lower, ci_upper = np.percentile(auc_scores, [2.5, 97.5])
```

---

## 7. Cross-Validated Feature Selection

**Problem:** The model features were likely chosen based on p-values from `analyze.py`, which ran on the same dataset. Selecting features from the full data and then training on the same full data introduces optimism bias — the model looks better than it will perform on new patients.

**Fix:** Feature selection must happen *inside* the cross-validation loop, not before it. Use one of:

- **Option A (simple):** Apply a p-value filter *within each CV fold* using `SelectFdr` or manual filtering on the training fold only.
- **Option B (recommended):** Use `RFECV` (Recursive Feature Elimination with CV) — it selects the optimal feature subset while estimating performance simultaneously.

```python
from sklearn.feature_selection import RFECV

selector = RFECV(
    estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    step=1,
    cv=StratifiedKFold(5),
    scoring='roc_auc',
    min_features_to_select=3,
)
selector.fit(x, y)
optimal_features = [f for f, s in zip(feature_names, selector.support_) if s]
```

---

## 8. Check Events Per Variable (EPV)

**Problem:** Logistic regression requires approximately 10 AFib-positive cases per predictor variable (the EPV rule). With 8 features, you need ~80 AFib-positive patients in the training set. If your dataset is smaller, the model is unreliable.

**Action:** After loading data, add an explicit check and log it:

```python
n_positive = int(y.sum())
n_features = len(self.model_features)
epv = n_positive / n_features
# Log: f"EPV = {epv:.1f} (minimum recommended: 10)"
# Warn if EPV < 10
```

If EPV < 10, consider reducing the number of model features.

---

## 9. Calibration Check

**Problem:** A model can have good AUC (discrimination) but poor calibration — e.g., it says 70% risk when the true risk is 30%. For a clinical risk calculator, calibration matters: the predicted probability must mean something real.

**Fix:** Add a calibration plot (reliability diagram) and compute the Brier score. If calibration is poor, apply Platt scaling (logistic calibration) or isotonic regression on the held-out test set.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, model.predict_proba(x_test)[:, 1], n_bins=10
)
# Plot and inspect. If curves diverge significantly, apply:
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(x_test, y_test)
```

---

## Summary — Priority Order

| # | Step | Blocking for clinical use? |
|---|---|---|
| 1 | Fix continuous variable scorecard | Yes — current scorecard gives wrong scores |
| 2 | Stratified train/test split | Yes — no valid performance estimate without it |
| 3 | Cross-validation + metrics export | Yes — needed to quote any performance claim |
| 4 | Class imbalance (`class_weight='balanced'`) | Yes — consistency with analyze.py, reduces bias |
| 5 | Bootstrap CIs for metrics | Yes — required by any clinical reviewer |
| 6 | Cross-validated feature selection | High — prevents optimistic bias |
| 7 | EPV check | Medium — safety check, easy to add |
| 8 | Calibration check | Medium — important if quoting probabilities to clinicians |
