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
    Loads a pre-trained model and scorecard, then makes predictions on new patient data.
"""
import json
import numpy as np
from joblib import load
from utils.config_singleton import ConfigSingleton


def load_model_and_scorecard(
    mdl_path: str,
    score_card_path: str
    ) -> tuple:
    """
    Loads the trained model and scorecard from disk.
    
    Args:
        mdl_path (str): Path to the .joblib model file
        scorecard_path (str): Path to the .json scorecard file
    
    Returns:
        tuple: (model, scorecard) where scorecard is a dict containing intercept and points
    
    Raises:
        FileNotFoundError: If model or scorecard files not found
        json.JSONDecodeError: If scorecard file is not valid JSON
    """
    # Load model
    mdl = load(mdl_path)
    print(f"Model loaded from: {mdl_path}")

    # Load scorecard
    with open(score_card_path, 'r', encoding='utf-8') as f:
        score_card = json.load(f)
    print(f"Scorecard loaded from: {score_card_path}")

    return mdl, score_card


def predict_from_scorecard(
    score_card: dict,
    patient_dict: dict,
    ) -> dict:
    """
    Calculates risk score using the point-based scorecard.
    
    Args:
        score_card (dict): Scorecard dictionary with 'intercept' and 'variables' keys
        patient_dict (dict): Patient data dictionary with feature values
    
    Returns:
        dict: Risk score and individual points breakdown
    """
    intercept = score_card['intercept']
    variables = score_card['variables']

    # Calculate total points
    total_points = intercept
    points_breakdown = {}

    for f, points in variables.items():
        if f in patient_dict:
            points_breakdown[f] = points
            total_points += points
        else:
            print(f"WARNING: Feature '{f}' not found in patient data (skipped)")

    return {
        'total_points': total_points,
        'individual_points': points_breakdown,
        'intercept': intercept,
    }


def predict_with_model(
    mdl,
    patient_dict: dict,
    mdl_features: list,
    ) -> float:
    """
    Calculates risk probability using the trained sklearn logistic regression model.
    
    Args:
        mdl: Trained LogisticRegression model
        patient_dict (dict): Patient data dictionary with feature values
        mdl_features (list): List of feature names in correct order
    
    Returns:
        float: Risk probability (0-1)
    """
    # Prepare patient data in correct order
    x_patient = np.array([
        patient_dict[feature] for feature in mdl_features
    ]).reshape(1, -1)

    # Get probability prediction
    risk_probability = mdl.predict_proba(x_patient)[0][1]

    return risk_probability


if __name__ == "__main__":
    print("="*70)
    print("AFib Risk Score Prediction - Inference Pipeline")
    print("="*70)

    try:
        # Load configuration
        ConfigSingleton.set()
        CFG = ConfigSingleton.get()

        # Get model paths from config
        model_path = CFG.model.model_output.model_path
        scorecard_path = CFG.model.model_output.scorecard_path

        print(f"\nLoading model from: {model_path}")
        print(f"Loading scorecard from: {scorecard_path}")

        # Load model and scorecard
        model, scorecard = load_model_and_scorecard(model_path, scorecard_path)

        # Get model features from config
        model_features = CFG.model.model_features
        feature_count = len(model_features)

        print(f"\nModel loaded with {feature_count} features")
        print(
            f"Training data completeness: { \
                scorecard['training_data_info']['data_completeness_%']:.1f}%")

        # Create sample patient data (example values)
        print("\n" + "="*70)
        print("Sample Patient #1 - Typical Risk Profile")
        print("="*70)

        patient_1 = {
            "Enddiastolický rozměr levé komory (LVEDD)": 52.0,
            "Interventrikulární septum (IVS)": 12.5,
            "CHA₂DS₂-VASc": 3,
            "Ejekční frakce levé komory (LVEF)": 55.0,
            "BMI": 28.5,
            "Typ akutní ischemie (choice=Teritoriální)": 1,
            "Osobní anamnéza (choice=Hyperlipidémie)": 1,
            "Sérový kreatinin": 110.0,
        }

        print("\nPatient values:")
        for feature, value in patient_1.items():
            print(f"  - {feature.split('(', maxsplit=1)[0].strip()}: {value}")

        # Predict using model
        risk_prob_1 = predict_with_model(model, patient_1, model_features)

        # Calculate point score using scorecard
        score_1 = predict_from_scorecard(scorecard, patient_1)

        print("\nPrediction Results:")
        print(f"  - Risk Probability: {risk_prob_1:.4f} ({risk_prob_1*100:.2f}%)")
        print(f"  - Point-Based Score: {score_1['total_points']:.2f}")
        print(f"  - Intercept: {score_1['intercept']:.4f}")

        # Create another sample patient (lower risk)
        print("\n" + "="*70)
        print("Sample Patient #2 - Lower Risk Profile")
        print("="*70)

        patient_2 = {
            "Enddiastolický rozměr levé komory (LVEDD)": 45.0,
            "Interventrikulární septum (IVS)": 10.0,
            "CHA₂DS₂-VASc": 1,
            "Ejekční frakce levé komory (LVEF)": 65.0,
            "BMI": 25.0,
            "Typ akutní ischemie (choice=Teritoriální)": 0,
            "Osobní anamnéza (choice=Hyperlipidémie)": 0,
            "Sérový kreatinin": 90.0,
        }

        print("\nPatient values:")
        for feature, value in patient_2.items():
            print(f"  - {feature.split('(', maxsplit=1)[0].strip()}: {value}")

        # Predict
        risk_prob_2 = predict_with_model(model, patient_2, model_features)
        score_2 = predict_from_scorecard(scorecard, patient_2)

        print("\nPrediction Results:")
        print(f"  - Risk Probability: {risk_prob_2:.4f} ({risk_prob_2*100:.2f}%)")
        print(f"  - Point-Based Score: {score_2['total_points']:.2f}")
        print(f"  - Intercept: {score_2['intercept']:.4f}")

        # Create another sample patient (high risk)
        print("\n" + "="*70)
        print("Sample Patient #3 - High Risk Profile")
        print("="*70)

        patient_3 = {
            "Enddiastolický rozměr levé komory (LVEDD)": 65.0,
            "Interventrikulární septum (IVS)": 10.0,
            "CHA₂DS₂-VASc": 7,
            "Ejekční frakce levé komory (LVEF)": 35.0,
            "BMI": 35.0,
            "Typ akutní ischemie (choice=Teritoriální)": 1,
            "Osobní anamnéza (choice=Hyperlipidémie)": 1,
            "Sérový kreatinin": 90.0,
        }

        print("\nPatient values:")
        for feature, value in patient_3.items():
            print(f"  - {feature.split('(', maxsplit=1)[0].strip()}: {value}")

        # Predict
        risk_prob_3 = predict_with_model(model, patient_3, model_features)
        score_3 = predict_from_scorecard(scorecard, patient_3)

        print("\nPrediction Results:")
        print(f"  - Risk Probability: {risk_prob_3:.4f} ({risk_prob_3*100:.2f}%)")
        print(f"  - Point-Based Score: {score_3['total_points']:.2f}")
        print(f"  - Intercept: {score_3['intercept']:.4f}")

        # Summary comparison
        print("\n" + "="*70)
        print("Comparison Summary")
        print("="*70)
        print(f"Patient 1 Risk Probability: {risk_prob_1:.4f} vs Patient 2: {risk_prob_2:.4f} \
            vs Patient 3: {risk_prob_3:.4f}")
        # print(f"Risk Difference: {(risk_prob_1 - risk_prob_2)*100:.2f} percentage points")
        # print(f"Patient 1 is {'HIGHER' if risk_prob_1 > risk_prob_2 else 'LOWER'} risk")

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
