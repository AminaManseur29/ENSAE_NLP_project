"""
Module pour charger un modèle déjà entraîné et prédire à partir d'un texte.
Utilise les embeddings ou les features handcrafted.
"""

import os
import joblib
import numpy as np

from application.embeddings_definition_one_text import (
    compute_bert_embedding,
    compute_roberta_embedding,
    compute_sbert_embedding,
)
from application.features_definition_one_text import compute_handcrafted_features_one_text

# -------------------------------
# Charger modèle et scaler
# -------------------------------


def load_model_and_scaler(model_name, base_dir="outputs"):
    """
    Charge un modèle et son scaler associés depuis le dossier `outputs/`.

    Args:
        model_name (str): Nom du modèle (ex: "LogisticRegression_bert", "XGBoost_features").
        base_dir (str): Base directory.

    Returns:
        model, scaler
    """
    model_path = os.path.join(base_dir, "models", f"{model_name}_model.joblib")
    scaler_path = os.path.join(base_dir, "scalers", f"{model_name}_scaler.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle {model_path} introuvable.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler {scaler_path} introuvable.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"✅ Modèle {model_name} et scaler chargés.")

    return model, scaler

# -------------------------------
# Prédiction pour un seul texte
# -------------------------------


def predict_text_class(text, model_name, feature_type="bert"):
    """
    Prédit la classe d'un texte.

    Args:
        text (str): Texte d'entrée.
        model_name (str): Nom du modèle entraîné à utiliser.
        feature_type (str): "bert", "roberta", "sbert" ou "handcrafted".

    Returns:
        prediction (int): 0 = humain, 1 = LLM
    """

    model, scaler = load_model_and_scaler(model_name)

    # Choix des features
    if feature_type == "bert":
        features = compute_bert_embedding(text)
    elif feature_type == "roberta":
        features = compute_roberta_embedding(text)
    elif feature_type == "sbert":
        features = compute_sbert_embedding(text)
    elif feature_type == "handcrafted":
        features = compute_handcrafted_features_one_text(text)  # à définir
    else:
        raise ValueError(f"Feature type '{feature_type}' non reconnu.")

    features = np.array(features).reshape(1, -1)  # reshape pour sklearn

    # Normaliser
    features_scaled = scaler.transform(features)

    # Prédiction
    prediction = model.predict(features_scaled)[0]

    return prediction
