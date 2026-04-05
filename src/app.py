"""
FastAPI Inference Server
All configuration read from params.yaml.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import joblib
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from model_utils import load_model_bundle


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


PARAMS       = load_params()
API_CFG      = PARAMS["api"]
FEAT_NAMES   = list(PARAMS["streamlit"]["feature_info"].keys())
CLASS_NAMES  = PARAMS["data"]["class_names"]
MODEL_PATH   = os.getenv("MODEL_PATH",  API_CFG["model_path"])
SCALER_PATH  = os.getenv("SCALER_PATH", API_CFG["scaler_path"])

model      = None
scaler     = None
model_type = "unknown"


class PatientFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }})
    age: int      = Field(..., ge=1,   le=120)
    sex: int      = Field(..., ge=0,   le=1)
    cp: int       = Field(..., ge=0,   le=3)
    trestbps: int = Field(..., ge=50,  le=250)
    chol: int     = Field(..., ge=100, le=600)
    fbs: int      = Field(..., ge=0,   le=1)
    restecg: int  = Field(..., ge=0,   le=2)
    thalach: int  = Field(..., ge=50,  le=250)
    exang: int    = Field(..., ge=0,   le=1)
    oldpeak: float= Field(..., ge=0.0, le=10.0)
    slope: int    = Field(..., ge=0,   le=2)
    ca: int       = Field(..., ge=0,   le=4)
    thal: int     = Field(..., ge=0,   le=3)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_no_disease: float
    probability_disease: float
    model_type: str


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, model_type
    model, model_type, _ = load_model_bundle(MODEL_PATH)
    print(f"[startup] Model loaded: {model_type}  path='{MODEL_PATH}'")
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"[startup] Scaler loaded: '{SCALER_PATH}'")
    except FileNotFoundError:
        print(f"[startup] WARNING: Scaler not found at '{SCALER_PATH}'")
    yield
    print("[shutdown] Done.")


app = FastAPI(
    title=API_CFG["title"],
    version=API_CFG["version"],
    description="MLOps inference API for heart disease classification",
    lifespan=lifespan,
)


def _prepare(patient: PatientFeatures) -> np.ndarray:
    row = pd.DataFrame([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal,
    ]], columns=FEAT_NAMES)
    return scaler.transform(row) if scaler is not None else row.values


def _predict(patient: PatientFeatures) -> PredictionResponse:
    X    = _prepare(patient)
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    return PredictionResponse(
        prediction=pred,
        prediction_label=CLASS_NAMES[pred],
        probability_no_disease=round(float(prob[0]), 4),
        probability_disease=round(float(prob[1]), 4),
        model_type=model_type,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_type": model_type}


@app.get("/model_info")
def model_info():
    rp = PARAMS["training"]["results_output_path"]
    if Path(rp).exists():
        with open(rp) as f:
            return json.load(f)
    return {"model_type": model_type}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predict(patient)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return BatchResponse(predictions=[_predict(p) for p in batch.patients], total=len(batch.patients))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_CFG["host"], port=API_CFG["port"], reload=False)
