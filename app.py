from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from model_pipeline import load_model, retrain_and_save
from fastapi.responses import FileResponse

MODEL_PATH = "rf_model.joblib"


class WaterSample(BaseModel):
    """Données d'entrée pour une observation (1 ligne du dataset)."""

    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


class RetrainParams(BaseModel):
    """Hyperparamètres pour le ré-entraînement du RandomForest."""

    n_estimators: int = 300
    max_depth: int | None = None
    random_state: int = 42


app = FastAPI(
    title="Water Potability API",
    description="API FastAPI pour prédire la potabilité de l'eau et réentraîner un RandomForest.",
    version="1.0.0",
)

model = None  # sera chargé au démarrage


@app.on_event("startup")
def startup_event():
    """Chargement du modèle au lancement du serveur."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle : {e}")
        model = None


@app.get("/")
def root():
    return {
        "message": "Water Potability API is running.",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict (POST)",
            "retrain": "/retrain (POST)",
        },
    }


@app.post("/predict")
def predict_potability(sample: WaterSample):
    """
    Prédit si l'eau est potable (1) ou non potable (0)
    à partir des caractéristiques envoyées.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur.")

    try:
        data = np.array(
            [
                [
                    sample.ph,
                    sample.Hardness,
                    sample.Solids,
                    sample.Chloramines,
                    sample.Sulfate,
                    sample.Conductivity,
                    sample.Organic_carbon,
                    sample.Trihalomethanes,
                    sample.Turbidity,
                ]
            ]
        )

        pred = model.predict(data)[0]
        pred_int = int(pred)
        label = "potable" if pred_int == 1 else "non potable"

        return {"prediction": pred_int, "label": label}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {e}",
        )


@app.post("/retrain")
def retrain_model(params: RetrainParams):
    """
    Réentraîne le modèle RandomForest avec les paramètres fournis.
    Sauvegarde le nouveau modèle et recharge le modèle en mémoire.
    """
    global model
    try:
        model, (X_test, y_test) = retrain_and_save(
            data_path="water_potability.csv",
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
            model_path=MODEL_PATH,
        )

        # petite évaluation rapide
        y_pred = model.predict(X_test)
        accuracy = float((y_pred == y_test).mean())

        return {
            "status": "success",
            "message": "Modèle ré-entraîné et rechargé.",
            "new_accuracy": accuracy,
            "parameters_used": params.dict(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant le réentraînement : {e}",
        )
@app.get("/ui", response_class=FileResponse)
def get_ui():
    return FileResponse("index.html")
