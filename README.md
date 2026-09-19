# 🚗 Scoring de résiliation — Assurance Auto

Modèle de Machine Learning qui estime la probabilité qu'un client résilie son contrat d'assurance auto, avec une interface web permettant à un conseiller de saisir le profil d'un client et d'obtenir immédiatement son niveau de risque.

**Application en ligne :** _à compléter après déploiement (étape 28)_
**Modèle :** Random Forest (`class_weight='balanced'`) · AUC test ≈ 0,85

## Structure du projet

```
scoring_resiliation/
├── data/
│   └── dataset_assurance_ML.xlsx      # données brutes (500 clients, 27 colonnes)
├── models/
│   ├── pipeline_resiliation.pkl       # pipeline scikit-learn complet (prétraitement + modèle)
│   └── metadata.json                  # colonnes, bornes des curseurs, modalités des menus
├── notebooks/
│   └── tp_final.ipynb                 # exploration : choix des variables, entraînement, évaluation
├── train_model.py                     # script d'entraînement (recrée le .pkl et le .json)
├── app.py                             # interface Streamlit
├── requirements.txt
└── .streamlit/config.toml             # thème de l'application
```

## Lancer le projet en local

```bash
conda activate formation_ml
pip install -r requirements.txt

# Ré-entraîner le modèle (facultatif si models/ est déjà présent)
python train_model.py

# Lancer l'interface
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Variables utilisées par le modèle

- **Numériques :** Âge, Salaire Annuel, Prime Annuelle, Ancienneté, Coeff. Bonus-Malus, Nb Sinistres (3 ans), Montant Sinistres, Score Risque
- **Catégorielles :** Type Contrat, Catégorie Prof., Usage Véhicule, Dernier Sinistre

`Statut Contrat` est volontairement exclue : elle encode presque parfaitement la cible (fuite de données) et ne serait pas connue à l'avance pour un client encore actif.
