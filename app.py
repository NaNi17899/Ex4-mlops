"""
Module: app.py
Description: Production FastAPI service for serving real-time Telco Churn predictions
             with rigorous Pydantic input schema validation, /predict and /metrics endpoints.
"""

import os
import json
from typing import Literal, Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predictor import ChurnPredictor

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Automated MLOps Support Vector Machine (SVM) inference service with Pydantic validation.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of predictor
predictor: Optional[ChurnPredictor] = None

def get_predictor() -> ChurnPredictor:
    global predictor
    if predictor is None:
        model_path = "models/best_svm_model.joblib"
        prep_path = "models/preprocessor.joblib"
        if not os.path.exists(model_path) or not os.path.exists(prep_path):
            raise RuntimeError("Model or preprocessor artifact missing. Please run main.py first.")
        predictor = ChurnPredictor(model_path=model_path, preprocessor_path=prep_path)
    return predictor

# -------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------
class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender ('Male' or 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has partner ('Yes' or 'No')")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents ('Yes' or 'No')")
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company (>=0)")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Has phone service ('Yes' or 'No')")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple lines status")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service provider")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing opted ('Yes' or 'No')")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(..., description="Customer payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly recurring charge amount (>= 0)")
    TotalCharges: float = Field(..., ge=0.0, description="Total cumulative charge amount (>= 0)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 79.85,
                "TotalCharges": 159.70
            }
        }
    }

class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="0 for No Churn, 1 for Churn")
    churn_label: str = Field(..., description="'No' or 'Yes'")
    churn_probability: float = Field(..., description="Probability of churn [0.0 - 1.0]")
    risk_level: str = Field(..., description="'Low', 'Medium', or 'High'")

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Telco Customer Churn Prediction API is active.",
        "model": "Support Vector Machine (SVM)",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "metrics": "/metrics (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "service": "telco-churn-svm-api"}

@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """
    Returns the production model's validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
    """
    metrics_path = "models/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics metadata not found. Run main.py first.")
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    return metrics_data

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerInput):
    """
    Accepts validated customer JSON and returns real-time churn prediction.
    """
    try:
        pred_engine = get_predictor()
        result = pred_engine.predict_single(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
