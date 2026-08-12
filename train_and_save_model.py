# train_and_save_model.py
"""
Train a RandomForest classifier on the CSV dataset, compute rule confidences,
save model + metadata to `model_artifacts.joblib`.

Usage:
  python train_and_save_model.py --csv /path/to/synthetic_water_quality_dataset.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

FEATURES = [
    "E_coli",
    "Total_Coliforms",
    "Enterococci",
    "Turbidity",
    "Residual_Chlorine",
    "Nitrate"
]

# ---- Heuristic rule functions (operate on a DataFrame) ----
def cholera_rule_df(df):
    return (
        (df["E_coli"] >= 100) &
        (df["Total_Coliforms"] >= 500) &
        (df["Residual_Chlorine"] < 0.2) &
        (df["Turbidity"] >= 5)
    )

def diarrhea_rule_df(df):
    # any 2 of the 3 conditions
    conds = pd.DataFrame({
        "c1": df["E_coli"] >= 50,
        "c2": df["Turbidity"] >= 3,
        "c3": df["Residual_Chlorine"] < 0.5
    })
    return conds.sum(axis=1) >= 2

def typhoid_rule_df(df):
    return (
        (df["Total_Coliforms"] >= 200) &
        (df["Enterococci"] >= 10) &
        (df["Nitrate"] >= 10)
    )

def dysentery_rule_df(df):
    return (
        (df["E_coli"] >= 100) &
        (df["Enterococci"] >= 20) &
        (df["Residual_Chlorine"] < 0.3)
    )

def hepatitis_rule_df(df):
    return (
        (df["Turbidity"] >= 10) &
        (df["Residual_Chlorine"] < 0.5)
    )

RULES = [
    ("Cholera_rule", "Cholera", cholera_rule_df),
    ("Diarrhea_rule", "Diarrhea", diarrhea_rule_df),
    ("Typhoid_rule", "Typhoid", typhoid_rule_df),
    ("Dysentery_rule", "Dysentery", dysentery_rule_df),
    ("HepatitisA_rule", "HepatitisA", hepatitis_rule_df)
]

def compute_rule_confidences(df):
    result = {}
    for rule_name, disease_label, rule_fn in RULES:
        mask = rule_fn(df)
        n_matches = int(mask.sum())
        if n_matches == 0:
            conf = None
        else:
            correct = (df.loc[mask, "Disease_Risk"] == disease_label).sum()
            conf = float(correct) / float(n_matches)  # fraction 0-1
        result[rule_name] = {
            "disease": disease_label,
            "matches": n_matches,
            "confidence": conf  # None if no matches
        }
    return result

def main(csv_path, out_path="model_artifacts.joblib", random_state=42):
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)

    # Basic checks
    for f in FEATURES + ["Disease_Risk"]:
        if f not in df.columns:
            raise ValueError(f"Missing column in CSV: {f}")

    # Compute descriptive ranges to populate UI defaults later
    stats = df[FEATURES].describe().T[["min", "50%", "max"]].rename(columns={"50%": "median"})

    # Compute rule confidences from the dataset (empirical precision)
    print("Computing rule confidences...")
    rule_confidences = compute_rule_confidences(df)

    # Prepare X, y
    X = df[FEATURES].copy()
    y_raw = df["Disease_Risk"].astype(str).copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Model - RandomForest baseline
    print("Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=300, random_state=random_state, min_samples_leaf=5, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluation
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature importances
    importances = dict(zip(FEATURES, clf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # Save useful artifacts: model, label encoder, features, defaults, ranges, rule confidences, importances
    defaults = X.median().to_dict()
    ranges = {f: {"min": float(X[f].min()), "max": float(X[f].max())} for f in FEATURES}

    artifacts = {
        "model": clf,
        "label_encoder": le,
        "features": FEATURES,
        "defaults": defaults,
        "ranges": ranges,
        "rule_confidences": rule_confidences,
        "feature_importances": importances
    }

    print("Saving artifacts to:", out_path)
    joblib.dump(artifacts, out_path)
    print("Saved. Done.")

if __name__ == "__main__":
    csv_path = r"C:\Users\shree\OneDrive\Documents\SIH\synthetic_water_quality_dataset.csv"
    out_path = "model_artifacts.joblib"
    main(csv_path, out_path)

