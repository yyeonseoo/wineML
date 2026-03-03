"""
Wine Quality Prediction API.
Loads red/white models from ../models/red/ and ../models/white/.
POST /predict with wine_type (red|white), GET /samples?type=red|white.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(title="Wine Quality API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-type artifacts: red, white (regression: predict rating 0-10)
models = {}      # "red" | "white" -> model
scalers = {}     # "red" | "white" -> scaler
samples = {}     # "red" | "white" -> list of sample dicts
feature_order = None


def load_artifacts():
    global models, scalers, samples, feature_order
    for wine_type in ("red", "white"):
        d = MODELS_DIR / wine_type
        if not d.exists():
            continue
        models[wine_type] = joblib.load(d / "model.joblib")
        scalers[wine_type] = joblib.load(d / "scaler.joblib")
        with open(d / "samples.json", encoding="utf-8") as f:
            samples[wine_type] = json.load(f)
    for wine_type in ("red", "white"):
        d = MODELS_DIR / wine_type
        if d.exists():
            with open(d / "feature_order.json") as f:
                feature_order = json.load(f)
            break


@app.on_event("startup")
def on_startup():
    load_artifacts()


class PredictRequest(BaseModel):
    wine_type: str  # "red" | "white"
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

    class Config:
        extra = "forbid"


@app.post("/predict")
def predict(req: PredictRequest):
    wine_type = req.wine_type.lower()
    if wine_type not in ("red", "white"):
        raise HTTPException(status_code=400, detail="wine_type must be 'red' or 'white'")
    if wine_type not in models or wine_type not in scalers:
        raise HTTPException(status_code=503, detail=f"Model for '{wine_type}' not loaded")
    model = models[wine_type]
    scaler = scalers[wine_type]
    row = {
        "fixed acidity": req.fixed_acidity,
        "volatile acidity": req.volatile_acidity,
        "citric acid": req.citric_acid,
        "residual sugar": req.residual_sugar,
        "chlorides": req.chlorides,
        "free sulfur dioxide": req.free_sulfur_dioxide,
        "total sulfur dioxide": req.total_sulfur_dioxide,
        "density": req.density,
        "pH": req.ph,
        "sulphates": req.sulphates,
        "alcohol": req.alcohol,
    }
    X = scaler.transform(pd.DataFrame([row], columns=feature_order))
    rating_10 = float(model.predict(X)[0])
    # Vivino-style 1-5 scale: 0->1, 10->5
    rating_1_5 = round(1 + (rating_10 / 10.0) * 4, 2)
    return {
        "wine_type": wine_type,
        "rating": round(rating_10, 2),
        "rating_1_5": rating_1_5,
    }


@app.get("/samples")
def get_samples(type: str | None = Query(None, description="red or white; omit for both")):
    if not samples:
        raise HTTPException(status_code=503, detail="Samples not loaded")
    if type is None:
        return samples.get("red", []) + samples.get("white", [])
    t = type.lower()
    if t not in ("red", "white"):
        raise HTTPException(status_code=400, detail="type must be 'red' or 'white'")
    return samples.get(t, [])


@app.get("/feature_order")
def get_feature_order():
    if feature_order is None:
        raise HTTPException(status_code=503, detail="Feature order not loaded")
    return feature_order


# Serve static files (index.html) from api/static
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
