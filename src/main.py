import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path


model = None


def load_model():
    """Load the trained model from disk."""
    global model
    model_path = Path("models/best_model.pkl")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Train the model first using train.py"
        )

    model = joblib.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown."""
    # Startup event
    try:
        load_model()
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")

    yield  # App runs here

    # Shutdown event
    print("Shutting down...")


app = FastAPI(
    title="House Price Predictor",
    version="1.0",
    lifespan=lifespan,
)


class HouseFeatures(BaseModel):
    """Input features for house price prediction."""

    GrLivArea: float  # Above grade living area sqft
    BedroomAbvGr: int  # Bedrooms above grade
    FullBath: int
    YearBuilt: int
    Neighborhood: str
    OverallQual: int


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_price: float
    confidence_range: dict


@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = "loaded" if model is not None else "not loaded"
    return {"status": "healthy", "model_version": "1.0", "model": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    """Predict house price from input features."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    df = pd.DataFrame([features.dict()])
    log_price = model.predict(df)[0]
    price = np.expm1(log_price)  # Reverse log transform

    return {
        "predicted_price": round(price, 2),
        "confidence_range": {
            "low": round(price * 0.9, 2),
            "high": round(price * 1.1, 2),
        },
    }
