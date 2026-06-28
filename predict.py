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
    Inference script for predicting AFib risk using a trained clinical risk score model.
    Loads the calibrated model and scorecard, then makes predictions on new patient data.

    Run as a demo:
        python predict.py

    The demo trains the model on first run (if no saved model is found), then shows
    calibrated AFib risk scores for three example patients.
"""
import json
import os
import numpy as np
from joblib import load
from utils.config_singleton import ConfigSingleton


def load_model_and_scorecard(mdl_path: str, score_card_path: str) -> tuple:
    """
    Loads a trained model and scorecard from disk.

    Args:
        mdl_path: Path to the .joblib model file (use calibrated model for deployment)
        score_card_path: Path to the .json scorecard file

    Returns:
        (model, scorecard) — model is a fitted sklearn estimator; scorecard is a dict

    Raises:
        FileNotFoundError: If either file is missing
    """
    mdl = load(mdl_path)
    with open(score_card_path, 'r', encoding='utf-8') as f:
        score_card = json.load(f)
    return mdl, score_card


def predict_from_scorecard(score_card: dict, patient_dict: dict) -> dict:
    """
    Calculates the point-based risk score from the scorecard.

    Each feature contributes: points_per_unit * actual_patient_value.
    Works for both binary (0/1) and continuous variables.

    Args:
        score_card: Scorecard dict with 'intercept_scaled' and 'variables' keys.
                    variables[feature] = {'points_per_unit': float, 'type': str}
        patient_dict: Patient data with feature values

    Returns:
        dict with total_points, individual_points breakdown, intercept_scaled
    """
    intercept = score_card['intercept_scaled']
    variables = score_card['variables']

    total_points = intercept
    points_breakdown = {}

    for feature, info in variables.items():
        if feature in patient_dict:
            contribution = info['points_per_unit'] * patient_dict[feature]
            points_breakdown[feature] = round(contribution, 4)
            total_points += contribution
        else:
            print(f"WARNING: Feature '{feature}' not found in patient data (skipped)")

    return {
        'total_points': round(total_points, 4),
        'individual_points': points_breakdown,
        'intercept_scaled': intercept,
    }


def risk_category(probability: float, prevalence: float) -> dict:
    """
    Maps a calibrated probability to a risk category relative to cohort prevalence.

    Thresholds (multiples of cohort prevalence):
        < 1×     → Nízké        — FiS nepravděpodobná
        1–1.5×   → Střední      — Mírně zvýšené riziko
        1.5–2×   → Vysoké       — Výrazně zvýšené riziko
        > 2×     → Velmi vysoké — Silné doporučení k monitoringu

    Args:
        probability: Calibrated AFib probability (0–1)
        prevalence:  Cohort AFib prevalence (from scorecard test_performance)

    Returns:
        dict with category, description, ratio_to_prevalence
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


def predict_with_model(mdl, patient_dict: dict, mdl_features: list) -> float:
    """
    Returns the AFib risk probability from a fitted sklearn model.

    Pass the calibrated model (risk_model_calibrated.joblib) to get
    probabilities that reflect true prevalence (~6%), not inflated values.

    Args:
        mdl: Fitted sklearn estimator (base or calibrated LogisticRegression)
        patient_dict: Patient data dictionary with feature values
        mdl_features: Feature names in the order the model expects them

    Returns:
        float: Risk probability in [0, 1]
    """
    x_patient = np.array([
        patient_dict[feature] for feature in mdl_features
    ]).reshape(1, -1)

    return float(mdl.predict_proba(x_patient)[0][1])


if __name__ == "__main__":
    print("=" * 70)
    print("AFib Risk Score — Demo")
    print("ICRC / Brno University of Technology")
    print("=" * 70)

    ConfigSingleton.set()
    CFG = ConfigSingleton.get()

    model_path     = CFG.model.model_output.model_path
    cal_model_path = model_path.replace('.joblib', '_calibrated.joblib')
    scorecard_path = CFG.model.model_output.scorecard_path

    # Train and export if no saved model exists
    if not (os.path.exists(cal_model_path) and os.path.exists(scorecard_path)):
        print("\nNo saved model found — training now (first run only)...")
        from risk_model import RiskScoreGenerator
        gen = RiskScoreGenerator()
        gen.train()
        gen.export_scorecard()
        print("Model trained and exported.\n")
    else:
        print(f"\nLoading model from:     {cal_model_path}")
        print(f"Loading scorecard from: {scorecard_path}")

    # Load calibrated model + scorecard
    model, scorecard = load_model_and_scorecard(cal_model_path, scorecard_path)

    # Selected features come from the scorecard (RFECV chose these from the original 8)
    selected_features = list(scorecard['variables'].keys())
    all_features      = scorecard['training_data_info'].get('all_features', selected_features)

    HLP_NOTE = (
        "POZOR — záporný koeficient Hyperlipidémie je pravděpodobně důsledek "
        "léčby statiny (treatment paradox): pacienti s diagnostikovanou "
        "hyperlipidémií jsou s vysokou pravděpodobností na statinové terapii, "
        "která snižuje riziko FiS. Data STROCZE neobsahují informaci o konkrétní "
        "medikaci, takže tato interpretace je hypotéza — nelze ji v tomto datasetu "
        "přímo ověřit."
    )

    print(f"\nOriginal features considered: {len(all_features)}")
    print(f"Features selected by model:   {len(selected_features)}")
    for f in selected_features:
        ppu = scorecard['variables'][f]['points_per_unit']
        ftype = scorecard['variables'][f]['type']
        print(f"  • {f}  [{ftype}, {ppu:+.2f} pts/unit]")
        if "Hyperlipid" in f and ppu < 0:
            for line in HLP_NOTE.split(". "):
                line = line.strip().rstrip(".")
                if line:
                    print(f"      ⚠  {line}.")

    # Performance summary — threshold-independent metrics + NPV
    tp  = scorecard.get('test_performance', {})
    cal = scorecard.get('calibration', {})
    n_pos = tp.get('n_positive', 0)
    n_test = tp.get('n_test', 1)
    print(f"\nModel performance (held-out test set, n={n_test}, "
          f"z toho AFib+: {n_pos}):")
    print(f"  AUC-ROC:     {tp.get('auc_roc', 0):.3f}  "
          f"[{tp.get('auc_ci_lower', 0):.3f} – {tp.get('auc_ci_upper', 0):.3f}]  95% CI")
    print(f"  Brier score: {cal.get('brier_calibrated', 0):.4f}  (kalibrovaný model)")
    print(f"  NPV:         {cal.get('calibrated_npv', 0):.3f}  "
          f"— při kategorii Nízké je FiS vyloučena s {\
              cal.get('calibrated_npv', 0)*100:.0f}% jistotou")
    print(f"  Poznámka: threshold pro bin. rozhodnutí volí lékař — "
          f"kategorie (níže) jsou primárním výstupem")

    # -------------------------------------------------------------------------
    # Sample patients — feature names taken directly from the scorecard to
    # avoid any mismatch with the exact Czech column names in the dataset.
    # Values: CHA₂DS₂-VASc is an integer 0–9; binary features are 0 or 1.
    # -------------------------------------------------------------------------
    f_cha   = selected_features[0]   # CHA₂DS₂-VASc
    f_terit = selected_features[1]   # Typ akutní ischemie (choice=Teritoriální)
    f_hlp   = selected_features[2]   # Osobní anamnéza (choice=Hyperlipidémie)

    patients = [
        {
            "label": "Pacient 1 — nízké riziko",
            "desc":  "Mladší, lakunární infarkt, bez hyperlipidémie",
            "data":  {f_cha: 1, f_terit: 0, f_hlp: 0},
        },
        {
            "label": "Pacient 2 — střední riziko",
            "desc":  "Středního věku, teritoriální infarkt, bez hyperlipidémie",
            "data":  {f_cha: 3, f_terit: 0, f_hlp: 1},
        },
        {
            "label": "Pacient 3 — vysoké riziko",
            "desc":  "Starší, teritoriální infarkt, hyperlipidémie",
            "data":  {f_cha: 6, f_terit: 1, f_hlp: 1},
        },
    ]

    prevalence = tp.get('cohort_prevalence') or tp.get('n_positive', 22) / tp.get('n_test', 377)

    print("\n" + "=" * 70)
    print("Ukázkové predikce")
    print("=" * 70)

    results = []
    for p in patients:
        prob  = predict_with_model(model, p["data"], selected_features)
        score = predict_from_scorecard(scorecard, p["data"])
        cat   = risk_category(prob, prevalence)

        results.append({
            "label":    p["label"],
            "prob":     prob,
            "pts":      score["total_points"],
            "category": cat["category"],
            "desc":     cat["description"],
            "ratio":    cat["ratio_to_prevalence"],
        })

        print(f"\n{'─'*60}")
        print(f"  {p['label']}")
        print(f"  {p['desc']}")
        print(f"{'─'*60}")
        print(f"  Vstupy:")
        for feat, val in p["data"].items():
            short = feat.split("(")[0].strip()
            ftype = scorecard['variables'].get(feat, {}).get('type', 'continuous')
            label = ("ano" if val == 1 else "ne") if ftype == 'binary' else str(val)
            print(f"    • {short:<35} {label}")
        print()
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  RIZIKO FiS:  {cat['category']:<27}│")
        print(f"  │  {cat['description']:<41}│")
        print(f"  │  ({\
            cat['ratio_to_prevalence']:.1f}× kohortová prevalence {prevalence*100:.1f} %)   │")
        print(f"  └─────────────────────────────────────────┘")
        print(f"  Kalibrovaná pravděpodobnost: {prob*100:.1f} %")
        print(f"  Bodové skóre:                {score['total_points']:.1f} bodů")

    print(f"\n{'═'*70}")
    print("Souhrn")
    print(f"{'═'*70}")
    print(f"  {'Pacient':<32} {'Kategorie':<16} {'Pravděp.':>9}  {'×prevalence':>11}")
    print(f"  {'-'*32} {'-'*16} {'-'*9}  {'-'*11}")
    for r in results:
        print(
            f"  {r['label']:<32} {r['category']:<16} {r['prob']*100:>8.1f} %  {r['ratio']:>10.1f}×")

    print(f"\n  Kohortová prevalence FiS: {\
        prevalence*100:.1f} %  (n={tp.get('n_positive','?')}/{tp.get('n_test','?')})")
    print(f"  Kategorie jsou definovány jako násobky kohortové prevalence:")
    print(f"    < 1×   → Nízké  |  1–1.5× → Střední  |  1.5–2× → Vysoké  |  > 2× → Velmi vysoké")
    print()
