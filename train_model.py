"""
train_model.py — Entraîne le pipeline de scoring de résiliation et sauvegarde
le modèle (models/pipeline_resiliation.pkl) et ses métadonnées (models/metadata.json).

Usage :
    python train_model.py
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --- Chemins (robustes, que le script soit lancé depuis la racine ou ailleurs) ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Résiliation"
NUM_COLS = [
    "Âge", "Salaire Annuel (€)", "Prime Annuelle (€)", "Ancienneté (mois)",
    "Coeff. Bonus-Malus", "Nb Sinistres (3 ans)",
    "Montant Sinistres (€)", "Score Risque (0-100)",
]
CAT_COLS = ["Type Contrat", "Catégorie Prof.", "Usage Véhicule", "Dernier Sinistre"]


def main():
    # --- Partie A : chargement & variables ---
    df = pd.read_excel(DATA_DIR / "dataset_assurance_ML.xlsx")
    df.to_csv(DATA_DIR / "dataset_assurance_ML.csv", index=False, encoding="utf-8-sig")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    print(f"Dataset : {df.shape} | Taux de résiliation : {y.mean():.1%}")

    # --- Partie B : pipeline, entraînement, évaluation ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])

    candidats = {
        "Régression Logistique": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        ),
    }
    pipelines = {
        nom: Pipeline([("prep", preprocessor), ("model", algo)])
        for nom, algo in candidats.items()
    }

    for nom, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"CV {nom:22s} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline = pipelines["Random Forest"]
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test — Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

    # --- Partie C : sauvegarde ---
    joblib.dump(pipeline, MODELS_DIR / "pipeline_resiliation.pkl")

    meta = {
        "modele": "Random Forest",
        "auc_test": round(float(auc), 3),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_ranges": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()),
                "median": float(X[c].median())}
            for c in NUM_COLS
        },
        "cat_values": {c: sorted(X[c].unique().tolist()) for c in CAT_COLS},
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Modèle et métadonnées sauvegardés dans {MODELS_DIR}/")


if __name__ == "__main__":
    main()
