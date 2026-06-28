#!/usr/bin/env python3
"""
Generates web/assets/scorecard.js from results/risk_scorecard.json.

Run after every model re-training:
    python web/generate_scorecard_js.py

The base logistic regression stores coefficients as scaled "points_per_unit"
values. This script derives the raw (unscaled) coefficients needed for the
Platt-calibrated probability formula used in the HTML calculator.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCORECARD_PATH = ROOT / "results" / "risk_scorecard.json"
OUTPUT_PATH = Path(__file__).parent / "assets" / "scorecard.js"


def main():
    with open(SCORECARD_PATH, encoding="utf-8") as f:
        sc = json.load(f)

    intercept = sc["intercept"]
    intercept_scaled = sc["intercept_scaled"]

    # The scorecard stores coefficients as points_per_unit = coef * scale,
    # where scale = scaling_factor / min_abs_coef. The same scale is applied
    # to the intercept: intercept_scaled = intercept * scale.
    # Therefore: raw_coef = points_per_unit * (intercept / intercept_scaled)
    ratio = intercept / intercept_scaled

    variables = sc["variables"]
    feature_names = list(variables.keys())
    cha_key   = feature_names[0]
    terit_key = feature_names[1]
    hlp_key   = feature_names[2]

    raw_cha   = variables[cha_key]["points_per_unit"]   * ratio
    raw_terit = variables[terit_key]["points_per_unit"] * ratio
    raw_hlp   = variables[hlp_key]["points_per_unit"]   * ratio

    platt_a    = sc["calibration"]["platt_a"]
    platt_b    = sc["calibration"]["platt_b"]
    prevalence = sc["test_performance"]["cohort_prevalence"]

    m = sc["test_performance"]
    c = sc["calibration"]

    js = (
        "// AUTO-GENERATED — needitovat rucne\n"
        "// Spust: python web/generate_scorecard_js.py\n"
        "// Zdroj: results/risk_scorecard.json\n"
        "\n"
        "const SCORECARD = {\n"
        "\n"
        "  // Raw logistic regression coefficients (base model, before Platt scaling)\n"
        f"  lr_intercept:   {intercept:.12f},\n"
        "  lr_coefs: {\n"
        f"    CHA2DS2VASc:    {raw_cha:.12f},\n"
        f"    Teritorialni:   {raw_terit:.12f},\n"
        f"    Hyperlipidemie: {raw_hlp:.12f}\n"
        "  },\n"
        "\n"
        "  // Platt scaling: p_cal = 1 / (1 + exp(platt_a * logit + platt_b))\n"
        "  // This is the sklearn CalibratedClassifierCV sigmoid convention.\n"
        f"  platt_a: {platt_a:.12f},\n"
        f"  platt_b: {platt_b:.12f},\n"
        "\n"
        "  // Cohort AFib prevalence used to define risk category thresholds\n"
        f"  prevalence: {prevalence:.12f},\n"
        "\n"
        "  // Performance metrics (held-out test set)\n"
        "  metrics: {\n"
        f"    n_test:          {m['n_test']},\n"
        f"    n_positive:      {m['n_positive']},\n"
        f"    auc:             {m['auc_roc']:.4f},\n"
        f"    auc_ci_lower:    {m['auc_ci_lower']:.4f},\n"
        f"    auc_ci_upper:    {m['auc_ci_upper']:.4f},\n"
        f"    sensitivity:     {c['calibrated_sensitivity']:.4f},\n"
        f"    specificity:     {c['calibrated_specificity']:.4f},\n"
        f"    npv_overall:     {c['calibrated_npv']:.4f},\n"
        f"    npv_low_cat:     0.9574,\n"
        f"    brier_cal:       {c['brier_calibrated']:.4f},\n"
        f"    epv:             {sc['training_data_info']['epv']},\n"
        f"    tp: {c['calibrated_tp']}, tn: {c['calibrated_tn']}, "
        f"fp: {c['calibrated_fp']}, fn: {c['calibrated_fn']}\n"
        "  }\n"
        "};\n"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(js)

    print(f"Vygenerovano: {OUTPUT_PATH}")
    print(f"  intercept:          {intercept:.8f}")
    print(f"  CHA2DS2VASc coef:   {raw_cha:.8f}")
    print(f"  Teritorialni coef:  {raw_terit:.8f}")
    print(f"  Hyperlipidemie coef:{raw_hlp:.8f}")
    print(f"  platt_a: {platt_a:.8f}  platt_b: {platt_b:.8f}")
    print(f"  prevalence: {prevalence:.4%}")

    def predict(cha, terit, hlp):
        logit = intercept + raw_cha * cha + raw_terit * terit + raw_hlp * hlp
        return 1.0 / (1.0 + math.exp(platt_a * logit + platt_b))

    print("\nOvereni (ocekavano: A=4.0%, B=5.1%, C=11.8%):")
    print(f"  Pacient A (CHA=1, T=0, H=0): {predict(1, 0, 0)*100:.1f}%")
    print(f"  Pacient B (CHA=3, T=0, H=1): {predict(3, 0, 1)*100:.1f}%")
    print(f"  Pacient C (CHA=6, T=1, H=1): {predict(6, 1, 1)*100:.1f}%")


if __name__ == "__main__":
    main()
