import streamlit as st
from src.application_to_one_text.prediction import predict_text_class

# -------------------------------
# Titre
# -------------------------------

st.title("Détection de texte généré par IA 🤖 vs Humain ✍️")

st.markdown("""
Cette application prédit si un texte est écrit par un humain
ou généré par un modèle de langage (LLM).
""")

# -------------------------------
# Sidebar : choix du modèle
# -------------------------------

st.sidebar.header("Configuration")

# Choix du type de features
feature_type = st.sidebar.selectbox(
    "Quel type de features veux-tu utiliser ?",
    ("bert", "roberta", "sbert", "handcrafted")
)

# Choix du modèle associé
model_name = st.sidebar.selectbox(
    "Quel modèle d'entraînement veux-tu utiliser ?",
    (
        "LogisticRegression_bert",
        "LinearSVC_bert",
        "XGBoost_bert",
        "LogisticRegression_roberta",
        "LinearSVC_roberta",
        "XGBoost_roberta",
        "LogisticRegression_sbert",
        "LinearSVC_sbert",
        "XGBoost_sbert",
        "LogisticRegression_handcrafted",
        "LinearSVC_handcrafted",
        "XGBoost_handcrafted"
    )
)

# -------------------------------
# Zone de texte pour l'utilisateur
# -------------------------------

st.subheader("Entrez votre texte 👇")

user_text = st.text_area(
    "Texte à analyser",
    placeholder="Écrivez ou copiez-collez votre texte ici...",
    height=300
)

# Bouton pour lancer la prédiction
if st.button("Lancer la prédiction 🚀"):
    if user_text.strip() == "":
        st.warning("Merci d'entrer un texte avant de prédire.")
    else:
        with st.spinner("Analyse du texte en cours..."):
            prediction = predict_text_class(user_text, model_name, feature_type)

            # Affichage du résultat
            if prediction == 0:
                st.success("✅ Ce texte est probablement **écrit par un humain**.")
            else:
                st.error("🚨 Ce texte est probablement **généré par un LLM**.")
