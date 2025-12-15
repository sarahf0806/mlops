import pandas as pd
import numpy as np

from pathlib import Path
import joblib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn

# On force MLflow à utiliser la base SQLite locale
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WaterPotability")


# =============================
# 1. Préparation des données
# =============================
def prepare_data(csv_path: str = "water_potability.csv",
                 test_size: float = 0.2,
                 random_state: int = 42):
    """
    Charge le CSV, gère les valeurs manquantes et fait le split train/test.
    """

    df = pd.read_csv(csv_path)
    df["ph"] = df["ph"].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =============================
# 2. Entraînement du modèle + MLflow
# =============================
def train_model(X_train,
                y_train,
                n_estimators: int = 300,
                max_depth=None,
                random_state: int = 42):
    """
    Entraîne un RandomForest et log tout dans MLflow (run + params + métriques + modèle).
    """

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # accuracy train
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        mlflow.log_metric("train_accuracy", train_acc)

        # log modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"[MLflow] train_accuracy = {train_acc}")

        return model


# =============================
# 3. Évaluation
# =============================
def evaluate_model(model, X_test, y_test):
    """
    Affiche le rapport + log l'accuracy test dans MLflow (si un run est actif).
    """

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy :", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    try:
        mlflow.log_metric("test_accuracy", acc)
    except Exception:
        # Si on évalue en dehors d'un run MLflow (cas main.py evaluate), on ignore.
        pass

    return acc


# =============================
# 4. Sauvegarde modèle disque
# =============================
def save_model(model, path: str = "rf_model.joblib"):
    """
    Sauvegarde le modèle au format joblib et essaie de le log en artefact MLflow.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")

    try:
        mlflow.log_artifact(path)
    except Exception:
        # Si pas de run actif, on ignore.
        pass


# =============================
# 5. Chargement modèle
# =============================
def load_model(path: str = "rf_model.joblib"):
    """
    Charge un modèle sauvegardé.
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)

    return joblib.load(path)


# =============================
# 6. Réentraînement + sauvegarde (utilisé par l'API FastAPI)
# =============================
def retrain_and_save(
    data_path: str = "water_potability.csv",
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
    model_path: str = "rf_model.joblib",
):
    """
    Réentraîne un RandomForest avec les hyperparamètres fournis,
    log dans MLflow, sauvegarde le modèle sur disque et retourne
    (model, (X_test, y_test)) pour que l'API puisse calculer une accuracy.
    """

    X_train, X_test, y_train, y_test = prepare_data(data_path)

    # On réutilise train_model => crée un run MLflow avec les bons params
    model = train_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    # On sauvegarde sur disque (et potentiellement en artefact MLflow)
    save_model(model, model_path)

    return model, (X_test, y_test)
