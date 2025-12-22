from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import pandas as pd

from model_pipeline import load_model, retrain_and_save

MODEL_PATH = "rf_model.joblib"


# =============================
# Modèles Pydantic (VALIDATION)
# =============================
class WaterSample(BaseModel):
    ph: float = Field(..., ge=0, le=14)
    Hardness: float = Field(..., ge=0, le=500)
    Solids: float = Field(..., ge=0, le=50000)
    Chloramines: float = Field(..., ge=0, le=4)
    Sulfate: float = Field(..., ge=0, le=400)
    Conductivity: float = Field(..., ge=0, le=2000)
    Organic_carbon: float = Field(..., ge=0, le=28)
    Trihalomethanes: float = Field(..., ge=0, le=120)
    Turbidity: float = Field(..., ge=0, le=6)


class RetrainParams(BaseModel):
    n_estimators: int = Field(300, ge=10, le=2000)
    max_depth: int | None = Field(None, ge=1, le=100)
    random_state: int = 42


# =============================
# Application FastAPI
# =============================
app = FastAPI(
    title="Water Potability API",
    description="API FastAPI pour prédire la potabilité de l'eau et réentraîner un RandomForest.",
    version="1.0.0",
)

model = None


# =============================
# Startup
# =============================
@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Modèle non chargé : {e}")
        model = None


# =============================
# Routes
# =============================
@app.get("/")
def root():
    return {
        "message": "Water Potability API is running.",
        "endpoints": ["/docs", "/predict", "/retrain", "/ui"],
    }


@app.post("/predict")
def predict_potability(sample: WaterSample):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur.")

    # 🔒 RÈGLES MÉTIER OMS (AVANT ML)
    if sample.ph < 6.5 or sample.ph > 8.5:
        return {
            "prediction": 0,
            "label": "non potable",
            "reason": "pH hors normes OMS (6.5 – 8.5)",
        }

    if sample.Turbidity > 1:
        return {
            "prediction": 0,
            "label": "non potable",
            "reason": "Turbidité trop élevée (> 1 NTU)",
        }

    # ✅ Création DataFrame avec noms de features
    df = pd.DataFrame(
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
        ],
        columns=[
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ],
    )

    try:
        pred = int(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return {
        "prediction": pred,
        "label": "potable" if pred == 1 else "non potable",
    }


@app.post("/retrain")
def retrain_model(params: RetrainParams):
    global model
    try:
        model, (X_test, y_test) = retrain_and_save(
            data_path="water_potability.csv",
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
            model_path=MODEL_PATH,
        )

        accuracy = float((model.predict(X_test) == y_test).mean())

        return {
            "status": "success",
            "new_accuracy": accuracy,
            "parameters_used": params.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur ré-entraînement : {e}")


@app.get("/ui", response_class=FileResponse)
def get_ui():
    return FileResponse("index.html")


# =============================
# Gestion erreurs validation
# =============================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": err["loc"][-1], "message": err["msg"]} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"errors": errors})
