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

# =============================
# Configuration MLflow
# =============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("WaterPotability")


# =============================
# 1. Préparation des données
# =============================
def prepare_data(
    csv_path: str = "water_potability.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Charge le CSV, gère les valeurs manquantes et fait le split train/test.
    """
    df = pd.read_csv(csv_path)

    # Remplacer ph = 0 par NaN
    df["ph"] = df["ph"].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# =============================
# 2. Entraînement du modèle
# =============================
def train_model(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
):
    """
    Entraîne un RandomForest et log les infos dans MLflow.
    """
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        mlflow.log_metric("train_accuracy", train_acc)

        # Log du modèle avec input_example pour éviter les warnings MLflow
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.iloc[:5],
        )

        print(f"[MLflow] train_accuracy = {train_acc:.4f}")

        return model


# =============================
# 3. Évaluation
# =============================
def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle et log l'accuracy test si un run MLflow est actif.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy :", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    try:
        mlflow.log_metric("test_accuracy", acc)
    except Exception as e:
        print(f"[MLflow] Metric logging skipped: {e}")

    return acc


# =============================
# 4. Sauvegarde modèle
# =============================
def save_model(model, path: str = "rf_model.joblib"):
    """
    Sauvegarde le modèle sur disque et tente un log MLflow.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")

    try:
        mlflow.log_artifact(path)
    except Exception as e:
        print(f"[MLflow] Artifact logging skipped: {e}")


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
# 6. Réentraînement + sauvegarde
# =============================
def retrain_and_save(
    data_path: str = "water_potability.csv",
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
    model_path: str = "rf_model.joblib",
):
    """
    Réentraîne le modèle, log MLflow, sauvegarde et retourne le modèle + données test.
    """
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    model = train_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    save_model(model, model_path)

    return model, (X_test, y_test)
