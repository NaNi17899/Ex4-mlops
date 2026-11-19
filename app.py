"""
Module: app.py
Description: Production FastAPI service for serving real-time Telco Churn predictions
             with rigorous Pydantic input schema validation, /predict, /metrics,
             /random-sample endpoints, and a luxury enterprise UI dashboard.
"""

import os
import json
import pandas as pd
from typing import Literal, Optional, List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
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
telco_df: Optional[pd.DataFrame] = None

def get_predictor() -> ChurnPredictor:
    global predictor
    if predictor is None:
        model_path = "models/best_svm_model.joblib"
        prep_path = "models/preprocessor.joblib"
        if not os.path.exists(model_path) or not os.path.exists(prep_path):
            raise RuntimeError("Model or preprocessor artifact missing. Please run main.py first.")
        predictor = ChurnPredictor(model_path=model_path, preprocessor_path=prep_path)
    return predictor

def get_dataset() -> pd.DataFrame:
    global telco_df
    if telco_df is None:
        csv_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        telco_df = pd.read_csv(csv_path)
    return telco_df

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
            "ui": "/ui (Interactive Web App)",
            "predict": "/predict (POST)",
            "random_sample": "/random-sample (GET)",
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

@app.get("/random-sample", tags=["Inference"])
def get_random_sample():
    """
    Pulls a genuine random customer row from the Telco dataset,
    including ground truth churn and customerID for comparison.
    """
    try:
        df = get_dataset()
        sample = df.sample(n=1).iloc[0].to_dict()
        
        # Parse TotalCharges safely
        tc_raw = str(sample.get("TotalCharges", "")).strip()
        try:
            tc = float(tc_raw) if tc_raw != "" else float(sample.get("MonthlyCharges", 0.0))
        except ValueError:
            tc = float(sample.get("MonthlyCharges", 0.0))
            
        return {
            "customerID": str(sample.get("customerID", "CUST-DEMO")),
            "actual_churn": str(sample.get("Churn", "No")),
            "gender": str(sample.get("gender", "Female")),
            "SeniorCitizen": int(sample.get("SeniorCitizen", 0)),
            "Partner": str(sample.get("Partner", "No")),
            "Dependents": str(sample.get("Dependents", "No")),
            "tenure": int(sample.get("tenure", 1)),
            "PhoneService": str(sample.get("PhoneService", "Yes")),
            "MultipleLines": str(sample.get("MultipleLines", "No")),
            "InternetService": str(sample.get("InternetService", "Fiber optic")),
            "OnlineSecurity": str(sample.get("OnlineSecurity", "No")),
            "OnlineBackup": str(sample.get("OnlineBackup", "No")),
            "DeviceProtection": str(sample.get("DeviceProtection", "No")),
            "TechSupport": str(sample.get("TechSupport", "No")),
            "StreamingTV": str(sample.get("StreamingTV", "No")),
            "StreamingMovies": str(sample.get("StreamingMovies", "No")),
            "Contract": str(sample.get("Contract", "Month-to-month")),
            "PaperlessBilling": str(sample.get("PaperlessBilling", "Yes")),
            "PaymentMethod": str(sample.get("PaymentMethod", "Electronic check")),
            "MonthlyCharges": round(float(sample.get("MonthlyCharges", 50.0)), 2),
            "TotalCharges": round(tc, 2)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sampling dataset: {str(e)}"
        )

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

@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
def interactive_ui():
    """
    Serves a luxury enterprise dashboard to test real-time predictions with
    interactive sliders, one-click random customer sampling, and ground truth validation.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Churn Intelligence • Enterprise MLOps</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #070b14;
            --surface: #0f172a;
            --surface-card: rgba(17, 24, 39, 0.7);
            --surface-hover: rgba(30, 41, 59, 0.8);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --accent: #8b5cf6;
            --cyan: #06b6d4;
            --danger: #f43f5e;
            --danger-glow: rgba(244, 63, 94, 0.3);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.3);
            --warning: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-sub: #64748b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.08) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            padding: 24px 16px;
            -webkit-font-smoothing: antialiased;
        }

        .container { max-width: 1240px; margin: 0 auto; }

        /* Top Hero Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-bottom: 6px;
        }
        .badge-pill .pulse-dot {
            width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
            box-shadow: 0 0 8px #22c55e;
        }

        .hero-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub { color: var(--text-muted); font-size: 13px; margin-top: 3px; }

        .header-actions { display: flex; gap: 10px; align-items: center; }
        .header-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 14px;
            border-radius: 8px;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border);
            color: #cbd5e1;
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
        }
        .header-btn:hover {
            background: rgba(51, 65, 85, 0.8);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        /* KPI Stat Ribbon */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: var(--surface-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease;
        }
        .kpi-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.15); }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--primary), transparent);
        }
        .kpi-label { font-size: 11px; text-transform: uppercase; font-weight: 600; color: var(--text-sub); letter-spacing: 0.05em; }
        .kpi-val { font-size: 22px; font-weight: 800; color: #fff; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
        .kpi-tag { font-size: 11px; color: #38bdf8; font-weight: 500; margin-top: 2px; }

        /* Main Workspace Grid */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1.45fr 1fr;
            gap: 24px;
        }
        @media(max-width: 950px) { .workspace-grid { grid-template-columns: 1fr; } }

        /* Card container */
        .glass-panel {
            background: var(--surface-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }
        .panel-title { font-size: 16px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }

        /* Action Toolbar */
        .action-bar {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        @media(max-width: 550px) { .action-bar { grid-template-columns: 1fr; } }

        .btn-magic {
            cursor: pointer;
            border: none;
            border-radius: 10px;
            padding: 12px 18px;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            color: #fff;
        }

        .btn-random {
            background: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%);
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35);
        }
        .btn-random:hover {
            box-shadow: 0 6px 20px rgba(217, 70, 239, 0.5);
            transform: translateY(-2px);
        }
        .btn-random:active { transform: translateY(0); }

        .btn-predict {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
        }
        .btn-predict:hover {
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
            transform: translateY(-2px);
        }

        .preset-row {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
        }
        .btn-preset {
            flex: 1;
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-preset:hover {
            background: rgba(51, 65, 85, 0.7);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.15);
        }

        /* Form Sections */
        .section-label {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #818cf8;
            margin: 16px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .input-group label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
        }
        .input-control {
            background: #090e1a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 12px;
            color: #fff;
            font-size: 13px;
            font-weight: 500;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .input-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        /* Right Panel: Results */
        .customer-pill {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 16px;
            font-size: 12px;
        }
        .customer-id {
            font-family: 'JetBrains Mono', monospace;
            color: #38bdf8;
            font-weight: 600;
        }

        /* Ground Truth Match Banner */
        .gt-banner {
            display: none;
            padding: 12px 14px;
            border-radius: 10px;
            margin-bottom: 16px;
            font-size: 12px;
            font-weight: 600;
            animation: fadeIn 0.3s ease;
        }
        .gt-match {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #6ee7b7;
        }
        .gt-mismatch {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #fde68a;
        }

        /* Radial Gauge Card */
        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 24px 0 14px 0;
            text-align: center;
        }

        .circular-chart {
            display: block;
            margin: 0 auto;
            max-width: 170px;
            max-height: 170px;
        }
        .circle-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.08);
            stroke-width: 3.5;
        }
        .circle {
            fill: none;
            stroke-width: 3.8;
            stroke-linecap: round;
            transition: stroke-dasharray 0.8s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.5s ease;
        }
        .percentage {
            fill: #ffffff;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 800;
            font-size: 0.5em;
            text-anchor: middle;
        }

        .verdict-tag {
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 6px 16px;
            border-radius: 9999px;
            display: inline-block;
            margin-top: 14px;
        }
        .verdict-danger {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.4);
            box-shadow: 0 0 15px rgba(244, 63, 94, 0.2);
        }
        .verdict-success {
            background: rgba(16, 185, 129, 0.15);
            color: #4ade80;
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }

        /* Factor Breakdown Chips */
        .chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 16px 0;
        }
        .chip {
            font-size: 11px;
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 6px;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border);
            color: #94a3b8;
        }
        .chip-highlight {
            background: rgba(99, 102, 241, 0.15);
            border-color: rgba(99, 102, 241, 0.3);
            color: #c7d2fe;
        }

        /* Raw JSON toggle */
        details {
            background: #090e1a;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 14px;
            font-size: 12px;
        }
        summary { cursor: pointer; color: var(--text-muted); font-weight: 600; outline: none; }
        pre {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #67e8f9;
            margin-top: 8px;
            overflow-x: auto;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div>
                <div class="badge-pill">
                    <span class="pulse-dot"></span>
                    <span>Production MLOps Service • SVM Active</span>
                </div>
                <h1 class="hero-title">Telco Churn Intelligence</h1>
                <p class="hero-sub">Automated Support Vector Machine Inference & Dataset Testing Console</p>
            </div>
            <div class="header-actions">
                <a class="header-btn" href="/docs" target="_blank">⚡ Swagger API</a>
                <a class="header-btn" href="/metrics" target="_blank">📊 Metrics JSON</a>
                <a class="header-btn" href="http://localhost:8888/notebooks/Desktop/mlops%20external/Experiment_4_Telco_Churn_MLOps.ipynb?token=8b216ecbb5335d924265f8acbe217d76bcb0136ad4b83cd2" target="_blank">📓 Jupyter Notebook</a>
            </div>
        </header>

        <!-- KPI Telemetry Ribbon -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">ROC-AUC Score</div>
                <div class="kpi-val" style="color: #38bdf8;">0.8262</div>
                <div class="kpi-tag">+2.9% vs v1 Baseline</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Model Accuracy</div>
                <div class="kpi-val" style="color: #4ade80;">78.85%</div>
                <div class="kpi-tag">Stratified Test Set</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">F1-Score</div>
                <div class="kpi-val" style="color: #c084fc;">57.06%</div>
                <div class="kpi-tag">Class-Weighted</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Winning Model</div>
                <div class="kpi-val" style="font-size: 17px; margin-top: 6px;">Linear SVM</div>
                <div class="kpi-tag">Hyperparameter C = 0.1</div>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="workspace-grid">
            <!-- Left Panel: Interactive Inputs -->
            <div class="glass-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>🎛️ Customer Feature Profile</span>
                    </div>
                    <span style="font-size: 11px; color: var(--text-sub);">Pydantic Validated</span>
                </div>

                <!-- Primary Action Toolbar -->
                <div class="action-bar">
                    <button type="button" class="btn-magic btn-random" onclick="fetchRandomSample()">
                        <span>🎲 Random from Dataset</span>
                    </button>
                    <button type="button" class="btn-magic btn-predict" onclick="runPrediction()">
                        <span>⚡ Run Prediction</span>
                    </button>
                </div>

                <div class="preset-row">
                    <button type="button" class="btn-preset" onclick="loadPreset('high')">⚠️ Load High Churn Risk</button>
                    <button type="button" class="btn-preset" onclick="loadPreset('low')">🛡️ Load Loyal Customer</button>
                </div>

                <form id="profileForm">
                    <!-- Section 1 -->
                    <div class="section-label">👤 Demographics & Contract Terms</div>
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Gender</label>
                            <select id="gender" class="input-control"><option>Female</option><option>Male</option></select>
                        </div>
                        <div class="input-group">
                            <label>Senior Citizen</label>
                            <select id="SeniorCitizen" class="input-control"><option value="0">No (0)</option><option value="1">Yes (1)</option></select>
                        </div>
                        <div class="input-group">
                            <label>Partner</label>
                            <select id="Partner" class="input-control"><option>Yes</option><option selected>No</option></select>
                        </div>
                        <div class="input-group">
                            <label>Dependents</label>
                            <select id="Dependents" class="input-control"><option>Yes</option><option selected>No</option></select>
                        </div>
                        <div class="input-group">
                            <label>Tenure (Months: 0-120)</label>
                            <input type="number" id="tenure" class="input-control" min="0" max="120" value="2">
                        </div>
                        <div class="input-group">
                            <label>Contract Type</label>
                            <select id="Contract" class="input-control">
                                <option>Month-to-month</option>
                                <option>One year</option>
                                <option>Two year</option>
                            </select>
                        </div>
                    </div>

                    <!-- Section 2 -->
                    <div class="section-label">🌐 Internet & Add-on Services</div>
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Internet Service</label>
                            <select id="InternetService" class="input-control">
                                <option>Fiber optic</option>
                                <option>DSL</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Phone Service</label>
                            <select id="PhoneService" class="input-control"><option>Yes</option><option>No</option></select>
                        </div>
                        <div class="input-group">
                            <label>Multiple Lines</label>
                            <select id="MultipleLines" class="input-control"><option>No</option><option>Yes</option><option>No phone service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Online Security</label>
                            <select id="OnlineSecurity" class="input-control"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Online Backup</label>
                            <select id="OnlineBackup" class="input-control"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Tech Support</label>
                            <select id="TechSupport" class="input-control"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Device Protection</label>
                            <select id="DeviceProtection" class="input-control"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Streaming TV</label>
                            <select id="StreamingTV" class="input-control"><option>Yes</option><option>No</option><option>No internet service</option></select>
                        </div>
                        <div class="input-group">
                            <label>Streaming Movies</label>
                            <select id="StreamingMovies" class="input-control"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                    </div>

                    <!-- Section 3 -->
                    <div class="section-label">💳 Billing & Payment</div>
                    <div class="form-grid">
                        <div class="input-group">
                            <label>Payment Method</label>
                            <select id="PaymentMethod" class="input-control">
                                <option>Electronic check</option>
                                <option>Mailed check</option>
                                <option>Bank transfer (automatic)</option>
                                <option>Credit card (automatic)</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label>Paperless Billing</label>
                            <select id="PaperlessBilling" class="input-control"><option>Yes</option><option>No</option></select>
                        </div>
                        <div class="input-group">
                            <label>Monthly Charges ($)</label>
                            <input type="number" step="0.05" id="MonthlyCharges" class="input-control" value="79.85">
                        </div>
                        <div class="input-group">
                            <label>Total Charges ($)</label>
                            <input type="number" step="0.05" id="TotalCharges" class="input-control" value="159.70">
                        </div>
                    </div>
                </form>
            </div>

            <!-- Right Panel: Prediction Results & Ground Truth -->
            <div class="glass-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>🎯 Real-Time Decision Engine</span>
                    </div>
                    <span style="font-size: 11px; color: #4ade80;">● Online</span>
                </div>

                <!-- Customer ID Pill -->
                <div class="customer-pill">
                    <span style="color: var(--text-muted);">Active Profile:</span>
                    <span id="activeCustId" class="customer-id">CUSTOM-USER</span>
                </div>

                <!-- Ground Truth Banner (Shown when Random is clicked) -->
                <div id="gtBanner" class="gt-banner">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span id="gtText">Dataset Ground Truth: YES</span>
                        <span id="gtBadge" style="padding: 2px 8px; border-radius: 4px; font-size: 10px;">ACCURATE</span>
                    </div>
                </div>

                <!-- Radial Probability Gauge -->
                <div class="gauge-container">
                    <svg viewBox="0 0 36 36" class="circular-chart">
                        <path class="circle-bg"
                            d="M18 2.0845
                              a 15.9155 15.9155 0 0 1 0 31.831
                              a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                        <path id="gaugeProgress" class="circle"
                            stroke="#f43f5e"
                            stroke-dasharray="0, 100"
                            d="M18 2.0845
                              a 15.9155 15.9155 0 0 1 0 31.831
                              a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                        <text id="gaugeText" x="18" y="20.35" class="percentage">--%</text>
                    </svg>

                    <div id="verdictBadge" class="verdict-tag verdict-danger">Awaiting Input</div>
                    <p id="verdictDesc" style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">
                        Click <strong>Random from Dataset</strong> or <strong>Run Prediction</strong>.
                    </p>
                </div>

                <!-- Contributing Risk Drivers -->
                <div class="section-label" style="margin-top: 10px;">🔍 Key Feature Influences</div>
                <div id="riskChips" class="chips-container">
                    <span class="chip">Awaiting inference...</span>
                </div>

                <!-- JSON Response Inspector -->
                <details>
                    <summary>View Raw JSON Payload & Response</summary>
                    <pre id="jsonPayload">{}</pre>
                </details>
            </div>
        </div>
    </div>

    <script>
        let lastActualChurn = null;

        async function fetchRandomSample() {
            try {
                const btn = document.querySelector('.btn-random');
                btn.style.opacity = '0.7';
                btn.innerText = '🎲 Sampling...';

                const res = await fetch('/random-sample');
                if(!res.ok) throw new Error('Failed to sample dataset');
                const sample = await res.json();

                // Populate all fields
                document.getElementById('gender').value = sample.gender;
                document.getElementById('SeniorCitizen').value = sample.SeniorCitizen;
                document.getElementById('Partner').value = sample.Partner;
                document.getElementById('Dependents').value = sample.Dependents;
                document.getElementById('tenure').value = sample.tenure;
                document.getElementById('Contract').value = sample.Contract;
                document.getElementById('InternetService').value = sample.InternetService;
                document.getElementById('PaymentMethod').value = sample.PaymentMethod;
                document.getElementById('PhoneService').value = sample.PhoneService;
                document.getElementById('MultipleLines').value = sample.MultipleLines;
                document.getElementById('OnlineSecurity').value = sample.OnlineSecurity;
                document.getElementById('OnlineBackup').value = sample.OnlineBackup;
                document.getElementById('TechSupport').value = sample.TechSupport;
                document.getElementById('DeviceProtection').value = sample.DeviceProtection;
                document.getElementById('StreamingTV').value = sample.StreamingTV;
                document.getElementById('StreamingMovies').value = sample.StreamingMovies;
                document.getElementById('PaperlessBilling').value = sample.PaperlessBilling;
                document.getElementById('MonthlyCharges').value = sample.MonthlyCharges;
                document.getElementById('TotalCharges').value = sample.TotalCharges;

                document.getElementById('activeCustId').innerText = sample.customerID + ' (Kaggle Dataset)';
                lastActualChurn = sample.actual_churn;

                btn.style.opacity = '1';
                btn.innerText = '🎲 Random from Dataset';

                // Automatically trigger prediction to compare immediately
                await runPrediction();

            } catch (err) {
                console.error(err);
                alert('Error fetching random sample: ' + err);
                document.querySelector('.btn-random').innerText = '🎲 Random from Dataset';
            }
        }

        function loadPreset(type) {
            lastActualChurn = null;
            document.getElementById('gtBanner').style.display = 'none';

            if(type === 'high') {
                document.getElementById('activeCustId').innerText = 'PRESET-HIGH-RISK';
                document.getElementById('Contract').value = 'Month-to-month';
                document.getElementById('tenure').value = 1;
                document.getElementById('InternetService').value = 'Fiber optic';
                document.getElementById('PaymentMethod').value = 'Electronic check';
                document.getElementById('OnlineSecurity').value = 'No';
                document.getElementById('TechSupport').value = 'No';
                document.getElementById('MonthlyCharges').value = 89.90;
                document.getElementById('TotalCharges').value = 89.90;
            } else {
                document.getElementById('activeCustId').innerText = 'PRESET-LOYAL-RETENTION';
                document.getElementById('Contract').value = 'Two year';
                document.getElementById('tenure').value = 65;
                document.getElementById('InternetService').value = 'DSL';
                document.getElementById('PaymentMethod').value = 'Credit card (automatic)';
                document.getElementById('OnlineSecurity').value = 'Yes';
                document.getElementById('TechSupport').value = 'Yes';
                document.getElementById('MonthlyCharges').value = 42.50;
                document.getElementById('TotalCharges').value = 2762.50;
            }
            runPrediction();
        }

        async function runPrediction() {
            const payload = {
                gender: document.getElementById('gender').value,
                SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                Partner: document.getElementById('Partner').value,
                Dependents: document.getElementById('Dependents').value,
                tenure: parseInt(document.getElementById('tenure').value),
                PhoneService: document.getElementById('PhoneService').value,
                MultipleLines: document.getElementById('MultipleLines').value,
                InternetService: document.getElementById('InternetService').value,
                OnlineSecurity: document.getElementById('OnlineSecurity').value,
                OnlineBackup: document.getElementById('OnlineBackup').value,
                DeviceProtection: document.getElementById('DeviceProtection').value,
                TechSupport: document.getElementById('TechSupport').value,
                StreamingTV: document.getElementById('StreamingTV').value,
                StreamingMovies: document.getElementById('StreamingMovies').value,
                Contract: document.getElementById('Contract').value,
                PaperlessBilling: document.getElementById('PaperlessBilling').value,
                PaymentMethod: document.getElementById('PaymentMethod').value,
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if(!res.ok) throw new Error('Prediction API rejected payload');
                const data = await res.json();

                // 1. Update Radial Gauge
                const probVal = Math.round(data.churn_probability * 100);
                const gaugeProgress = document.getElementById('gaugeProgress');
                const gaugeText = document.getElementById('gaugeText');

                gaugeProgress.setAttribute('stroke-dasharray', `${probVal}, 100`);
                gaugeText.textContent = `${probVal}%`;

                // 2. Verdict Badge & Color Shift
                const verdictBadge = document.getElementById('verdictBadge');
                const verdictDesc = document.getElementById('verdictDesc');

                if (data.churn_prediction === 1) {
                    gaugeProgress.setAttribute('stroke', '#f43f5e');
                    verdictBadge.className = 'verdict-tag verdict-danger';
                    verdictBadge.innerText = 'HIGH CHURN PROBABILITY';
                    verdictDesc.innerText = 'Immediate customer retention outreach is strongly recommended.';
                } else {
                    gaugeProgress.setAttribute('stroke', '#10b981');
                    verdictBadge.className = 'verdict-tag verdict-success';
                    verdictBadge.innerText = 'RETENTION SECURE';
                    verdictDesc.innerText = 'Customer profile exhibits stable, long-term tenure patterns.';
                }

                // 3. Ground Truth Verification Banner
                const gtBanner = document.getElementById('gtBanner');
                if (lastActualChurn !== null) {
                    gtBanner.style.display = 'block';
                    const isMatch = (data.churn_label === lastActualChurn);
                    if (isMatch) {
                        gtBanner.className = 'gt-banner gt-match';
                        document.getElementById('gtText').innerHTML = `<strong>Ground Truth Match:</strong> Customer actually <strong>${lastActualChurn.toUpperCase() === 'YES' ? 'CHURNED' : 'STAYED'}</strong> in dataset.`;
                        document.getElementById('gtBadge').innerText = '✓ MODEL MATCH';
                        document.getElementById('gtBadge').style.background = '#059669';
                        document.getElementById('gtBadge').style.color = '#fff';
                    } else {
                        gtBanner.className = 'gt-banner gt-mismatch';
                        document.getElementById('gtText').innerHTML = `<strong>Dataset Reality:</strong> Actually <strong>${lastActualChurn.toUpperCase() === 'YES' ? 'CHURNED' : 'STAYED'}</strong> | Model Predicted: <strong>${data.churn_label}</strong>`;
                        document.getElementById('gtBadge').innerText = 'BOUNDARY CASE';
                        document.getElementById('gtBadge').style.background = '#d97706';
                        document.getElementById('gtBadge').style.color = '#fff';
                    }
                } else {
                    gtBanner.style.display = 'none';
                }

                // 4. Contributing Factor Chips
                const chipsContainer = document.getElementById('riskChips');
                chipsContainer.innerHTML = '';
                const chips = [];

                if(payload.Contract === 'Month-to-month') chips.push('⚠️ Month-to-month Contract (+Risk)');
                else chips.push('🛡️ Long-term Contract (-Risk)');

                if(payload.InternetService === 'Fiber optic') chips.push('⚡ Fiber Optic (+Tech)');
                else if(payload.InternetService === 'No') chips.push('📦 No Internet');

                if(payload.tenure < 6) chips.push('⏳ New Customer (<6 mos)');
                else if(payload.tenure > 36) chips.push('💎 Veteran Customer (>3 yrs)');

                if(payload.PaymentMethod === 'Electronic check') chips.push('💳 Electronic Check');

                chips.forEach(text => {
                    const el = document.createElement('span');
                    el.className = text.includes('⚠️') ? 'chip chip-highlight' : 'chip';
                    el.innerText = text;
                    chipsContainer.appendChild(el);
                });

                // 5. Raw JSON Inspector
                document.getElementById('jsonPayload').textContent = JSON.stringify({
                    input: payload,
                    prediction: data
                }, null, 2);

            } catch (err) {
                alert('Prediction Error: ' + err);
            }
        }

        // Run initial prediction on load
        window.addEventListener('DOMContentLoaded', () => {
            runPrediction();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
