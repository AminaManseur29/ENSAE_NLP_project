import time
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------------------------
# Fonction principale d'entraînement et tuning
# -----------------------------------------------


def train_and_tune_models(x_train, y_train, x_test, y_test, feature_type):
    """
    Entraîne et tune 3 modèles classiques (LogReg, LinearSVC, XGBoost)
    sur les données x_train / y_train
    et évalue sur x_test / y_test.
    feature_type : str (ex: 'bert', 'roberta', 'sbert', 'bow', 'tfidf', 'handcrafted')
    """
    models_and_params = {
        f"LogisticRegression_{feature_type}": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        ),
        f"LinearSVC_{feature_type}": (
            LinearSVC(max_iter=10000, random_state=42),
            {
                "C": [0.01, 0.1, 1, 10],
                "loss": ["squared_hinge"]
            }
        ),
        f"XGBoost_{feature_type}": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1]
            }
        )
    }

    # Dossier output
    os.makedirs('outputs/models', exist_ok=True)

    results = []

    for name, (model, params) in models_and_params.items():
        print(f"\nTraining and tuning {name}...")
        start_time = time.time()

        grid_search = GridSearchCV(
            model,
            param_grid=params,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        training_time = time.time() - start_time

        print(f"Best params for {name}: {grid_search.best_params_}")
        print(f"Training time: {training_time:.2f} seconds")

        # Évaluation
        y_pred = best_model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        results.append({
            "Model": name,
            "Best Params": grid_search.best_params_,
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Training Time (s)": training_time
        })

        # Sauvegarde du modèle
        model_path = f"outputs/models/{name}_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"{name} enregistré dans {model_path} ✅")

    # Résultats finaux
    results_df = pd.DataFrame(results)
    return results_df
