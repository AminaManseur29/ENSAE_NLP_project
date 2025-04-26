import os
import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_and_evaluate_models(
    model_names,
    x_train,
    y_train,
    x_test,
    y_test,
    output_dir="outputs/",
    save_models=True
):
    """
    Entraîne et évalue une liste de modèles spécifiés.

    Args:
        model_names (list or str): Liste des modèles ou un seul modèle 
        ("LogisticRegression", "LinearSVC", "XGBoost").
        x_train (array): Données d'entraînement.
        y_train (array): Labels d'entraînement.
        x_test (array): Données de test.
        y_test (array): Labels de test.
        output_dir (str): Dossier pour sauvegarder les modèles.
        save_models (bool): Si True, sauvegarde les modèles entraînés.

    Returns:
        DataFrame avec les scores des modèles.
    """

    # Si on passe un seul modèle en string, on transforme en liste
    if isinstance(model_names, str):
        model_names = [model_names]

    # Définir les modèles disponibles
    all_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(max_iter=10000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Garde seulement les modèles demandés
    models = {name: all_models[name] for name in model_names}

    # Créer le dossier de sauvegarde si besoin
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for name, model in models.items():
        print(f"Training {name}...")

        start_time = time.time()

        model.fit(x_train, y_train)

        training_time = time.time() - start_time
        print(f"Training time for {name}: {training_time:.2f} seconds")

        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Training Time (s)": training_time
        })

        if save_models:
            model_path = os.path.join(output_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            print(f"{name} enregistré dans {model_path} ✅")

    return pd.DataFrame(results)
