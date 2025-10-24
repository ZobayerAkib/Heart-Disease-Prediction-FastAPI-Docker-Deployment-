# main.py
from fastapi import FastAPI
from functools import lru_cache
from transformers import pipeline
import time, redis, json, hashlib
from app.schemas import HeartData, PredictionOutput
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# ===============================
# Load Model and Scaler
# ===============================
model = joblib.load("model/heart_model.joblib")
scaler = joblib.load("model/scaler.joblib")

# List of model features
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# ===============================
# Endpoint: GET /health
# ===============================
@app.get("/health")
def health_check():
    """Check if API is running"""
    return {"status": "ok", "message": "Heart Disease Prediction API is healthy."}


# ===============================
# Endpoint: GET /info
# ===============================
@app.get("/info")
def model_info():
    """Return model information"""
    return {
        "model_type": type(model).__name__,
        "feature_count": len(FEATURES),
        "features": FEATURES
    }


# ===============================
# Endpoint: POST /predict
# ===============================
@app.post("/predict", response_model=PredictionOutput)
def predict_heart_disease(data: HeartData):
    """Predict whether a person has heart disease"""
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ]])

    # Scale input and make prediction
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Return as boolean: True = heart disease present, False = no disease
    return {"heart_disease": bool(prediction)}


# ----------------------------------------------------------------
# 🧠 Redis Setup (for distributed caching across multiple servers)
# ----------------------------------------------------------------
try:
    r = redis.Redis(host="redis", port=6379, db=0)
    r.ping()  # check if redis is running
    print("✅ Connected to Redis")
except Exception as e:
    r = None
    print("⚠️ Redis not available:", e)