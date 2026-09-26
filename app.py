"""
app.py — Interface Streamlit pour le scoring de résiliation client.

Lancement local : streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Scoring Résiliation", page_icon="🚗", layout="wide")


@st.cache_resource
def charger_modele():
    pipeline = joblib.load(MODELS_DIR / "pipeline_resiliation.pkl")
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, meta = charger_modele()
num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
rng, cats = meta["num_ranges"], meta["cat_values"]

st.title("🚗 Scoring de résiliation — Assurance Auto")
st.caption(f"Modèle : {meta['modele']} · AUC test : {meta['auc_test']} · 12 variables d'entrée")

# --- Barre latérale : profil du client ---
st.sidebar.header("👤 Profil du client")


def curseur(col, step=1.0, fmt=None):
    r = rng[col]
    val = st.sidebar.slider(
        col, min_value=r["min"], max_value=r["max"],
        value=r["median"], step=step, format=fmt,
    )
    return int(val) if fmt == "%d" else val


client = {}
client["Âge"] = curseur("Âge", 1.0, "%d")
client["Salaire Annuel (€)"] = curseur("Salaire Annuel (€)", 500.0, "%d")
client["Prime Annuelle (€)"] = curseur("Prime Annuelle (€)", 10.0, "%d")
client["Ancienneté (mois)"] = curseur("Ancienneté (mois)", 1.0, "%d")
client["Coeff. Bonus-Malus"] = curseur("Coeff. Bonus-Malus", 0.01, "%.2f")
client["Nb Sinistres (3 ans)"] = curseur("Nb Sinistres (3 ans)", 1.0, "%d")
client["Montant Sinistres (€)"] = curseur("Montant Sinistres (€)", 100.0, "%d")
client["Score Risque (0-100)"] = curseur("Score Risque (0-100)", 1.0, "%d")

st.sidebar.markdown("---")
for col in cat_cols:
    client[col] = st.sidebar.selectbox(col, cats[col])

st.sidebar.markdown("---")
seuil_risque = st.sidebar.slider(
    "🎚️ Seuil d'alerte (risque)", min_value=0.30, max_value=0.70,
    value=0.50, step=0.01,
    help="Probabilité à partir de laquelle un client est classé « à risque ».",
)
seuil_modere = max(0.10, seuil_risque - 0.15)

# --- Zone principale : prédiction ---
st.write("Renseignez le profil du client dans le panneau de gauche, puis cliquez sur **Prédire** pour estimer son risque de résiliation.")

if st.button("🔮 Prédire", type="primary", width="stretch"):
    df_client = pd.DataFrame([client])[num_cols + cat_cols]
    proba = float(pipeline.predict_proba(df_client)[0, 1])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilité de résiliation", f"{proba:.0%}")
        if proba >= seuil_risque:
            st.error("⚠️ Client À RISQUE — action de rétention conseillée")
        elif proba >= seuil_modere:
            st.warning("🟠 Risque modéré — à surveiller")
        else:
            st.success("✅ Client fidèle — risque faible")

    with col2:
        st.write("Niveau de risque")
        st.progress(min(max(proba, 0.0), 1.0))
        st.write("Données envoyées au modèle :")
        st.dataframe(
            df_client.T.astype(str).rename(columns={0: "Valeur"}),
            width="stretch",
        )

    # --- Importance des variables ---
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        noms = pipeline.named_steps["prep"].get_feature_names_out()
        imp = (
            pd.Series(model.feature_importances_, index=noms)
            .sort_values(ascending=False)
            .head(8)
        )
        imp.index = [n.split("__", 1)[1] for n in imp.index]
        st.subheader("📊 Les 8 variables les plus influentes du modèle")
        st.bar_chart(imp)
else:
    st.info("👈 Ajustez le profil dans la barre latérale, puis cliquez sur Prédire.")
