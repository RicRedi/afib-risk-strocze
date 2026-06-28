// AUTO-GENERATED — needitovat rucne
// Spust: python web/generate_scorecard_js.py
// Zdroj: results/risk_scorecard.json

const SCORECARD = {

  // Raw logistic regression coefficients (base model, before Platt scaling)
  lr_intercept:   -0.782905801175,
  lr_coefs: {
    CHA2DS2VASc:    0.286985260094,
    Teritorialni:   0.268836589674,
    Hyperlipidemie: -0.260563698473
  },

  // Platt scaling: p_cal = 1 / (1 + exp(platt_a * logit + platt_b))
  // This is the sklearn CalibratedClassifierCV sigmoid convention.
  platt_a: -0.801886441512,
  platt_b: 2.775580518658,

  // Cohort AFib prevalence used to define risk category thresholds
  prevalence: 0.058355437666,

  // Performance metrics (held-out test set)
  metrics: {
    n_test:          377,
    n_positive:      22,
    auc:             0.6469,
    auc_ci_lower:    0.5434,
    auc_ci_upper:    0.7398,
    sensitivity:     0.9545,
    specificity:     0.3014,
    npv_overall:     0.9907,
    npv_low_cat:     0.9574,
    brier_cal:       0.0545,
    epv:             29.67,
    tp: 21, tn: 107, fp: 248, fn: 1
  }
};
