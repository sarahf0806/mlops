import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
from pathlib import Path

# 1. Chargement + préparation des données


def prepare_data(csv_path="water_potability.csv", test_size=0.2, random_state=42):

    df = pd.read_csv(csv_path)
    df["ph"] = df["ph"].replace(0, np.nan)

    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# 2. Entraînement du modèle


def train_model(X_train, y_train, random_state=42):

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


# 3. Évaluation du modèle


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy :", acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return acc


# 4. Sauvegarde du modèle


def save_model(model, save_path="rf_model.joblib"):

    path = Path(save_path)
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


# 5. Chargement du modèle sauvegardé


def load_model(load_path="rf_model.joblib"):

    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier modèle '{path}' n'existe pas.")

    model = joblib.load(path)
    print(f"Modèle chargé depuis : {path}")
    return model
