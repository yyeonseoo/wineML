"""
Wine Quality Prediction API.
Loads model and scaler from ../models/ and exposes POST /predict and GET /samples.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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

# Load artifacts at startup
model = None
scaler = None
feature_order = None
class_names = None
samples_list = None


def load_artifacts():
    global model, scaler, feature_order, class_names, samples_list
    model = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_order.json") as f:
        feature_order = json.load(f)
    with open(MODELS_DIR / "class_names.json") as f:
        class_names = json.load(f)
    with open(MODELS_DIR / "samples.json") as f:
        samples_list = json.load(f)


@app.on_event("startup")
def on_startup():
    load_artifacts()


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float  # API uses 'ph' to avoid symbol
    sulphates: float
    alcohol: float

    class Config:
        # allow both 'ph' and 'pH' if we add alias
        extra = "forbid"


@app.post("/predict")
def predict(features: WineFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Build vector in same order as feature_order (DataFrame avoids sklearn name warning)
    row = {
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.ph,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol,
    }
    X = scaler.transform(pd.DataFrame([row], columns=feature_order))
    pred_class = int(model.predict(X)[0])
    label = class_names[pred_class]
    out = {"class": label, "class_index": pred_class}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0].tolist()
        out["probabilities"] = {class_names[i]: probs[i] for i in range(len(class_names))}
    return out


@app.get("/samples")
def get_samples():
    if samples_list is None:
        raise HTTPException(status_code=503, detail="Samples not loaded")
    # Return as-is (raw values). Frontend uses keys that match CSV: "fixed acidity" etc.
    return samples_list


@app.get("/feature_order")
def get_feature_order():
    if feature_order is None:
        raise HTTPException(status_code=503, detail="Feature order not loaded")
    return feature_order


# Serve static files (index.html) from api/static
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
