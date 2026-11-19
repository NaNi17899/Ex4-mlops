"""
Module: app.py
Description: Production FastAPI service for serving real-time Telco Churn predictions
             with rigorous Pydantic input schema validation, /predict and /metrics endpoints.
"""

import os
import json
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

@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
def interactive_ui():
    """
    Serves an interactive web application to test real-time predictions and inspect model metrics.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Customer Churn Prediction - MLOps SVM</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 24px; }
        header h1 { font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 6px; }
        header p { color: var(--text-muted); font-size: 14px; }
        .nav-links { margin-top: 10px; display: flex; justify-content: center; gap: 16px; font-size: 13px; }
        .nav-links a { color: #60a5fa; text-decoration: none; border-bottom: 1px dashed #60a5fa; }
        .nav-links a:hover { color: #93c5fd; }
        
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
        @media(max-width: 850px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .card h2 { font-size: 18px; margin-bottom: 16px; font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
        
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
        .form-group { display: flex; flex-direction: column; gap: 4px; }
        label { font-size: 12px; font-weight: 500; color: var(--text-muted); }
        input, select { background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; color: #fff; font-size: 13px; outline: none; }
        input:focus, select:focus { border-color: var(--primary); }
        
        .btn-group { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
        button { cursor: pointer; border: none; border-radius: 6px; padding: 10px 16px; font-size: 13px; font-weight: 600; transition: all 0.2s; }
        .btn-primary { background: var(--primary); color: white; flex: 1; min-width: 140px; }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-preset { background: #334155; color: #e2e8f0; }
        .btn-preset:hover { background: #475569; }
        
        .result-box { display: none; text-align: center; margin-top: 16px; padding: 20px; border-radius: 8px; }
        .result-high { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger); }
        .result-low { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--success); }
        .result-title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
        .result-prob { font-size: 32px; font-weight: 800; margin: 10px 0; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; text-transform: uppercase; }
        .badge-danger { background: var(--danger); color: white; }
        .badge-success { background: var(--success); color: white; }
        
        .metrics-card { font-size: 13px; }
        .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
        .metric-label { color: var(--text-muted); }
        .metric-val { font-weight: 600; color: #38bdf8; }
        
        pre { background: #0b1120; padding: 10px; border-radius: 6px; font-size: 11px; overflow-x: auto; color: #a5f3fc; text-align: left; margin-top: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📡 Telco Customer Churn Predictor</h1>
            <p>Automated MLOps Support Vector Machine (SVM) Pipeline</p>
            <div class="nav-links">
                <a href="/docs" target="_blank">📖 Swagger API Docs</a>
                <a href="/metrics" target="_blank">📊 JSON Metrics</a>
                <a href="http://localhost:8888/notebooks/Desktop/mlops%20external/Experiment_4_Telco_Churn_MLOps.ipynb?token=8b216ecbb5335d924265f8acbe217d76bcb0136ad4b83cd2" target="_blank">📓 Open Jupyter Notebook</a>
            </div>
        </header>

        <div class="grid">
            <!-- Left: Form -->
            <div class="card">
                <h2>Customer Profile Inputs</h2>
                <form id="churnForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Gender</label>
                            <select id="gender"><option>Female</option><option>Male</option></select>
                        </div>
                        <div class="form-group">
                            <label>Senior Citizen</label>
                            <select id="SeniorCitizen"><option value="0">No (0)</option><option value="1">Yes (1)</option></select>
                        </div>
                        <div class="form-group">
                            <label>Partner</label>
                            <select id="Partner"><option>Yes</option><option selected>No</option></select>
                        </div>
                        <div class="form-group">
                            <label>Dependents</label>
                            <select id="Dependents"><option>Yes</option><option selected>No</option></select>
                        </div>
                        <div class="form-group">
                            <label>Tenure (Months)</label>
                            <input type="number" id="tenure" min="0" max="120" value="2">
                        </div>
                        <div class="form-group">
                            <label>Contract Type</label>
                            <select id="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select>
                        </div>
                        <div class="form-group">
                            <label>Internet Service</label>
                            <select id="InternetService"><option>Fiber optic</option><option>DSL</option><option>No</option></select>
                        </div>
                        <div class="form-group">
                            <label>Payment Method</label>
                            <select id="PaymentMethod">
                                <option>Electronic check</option>
                                <option>Mailed check</option>
                                <option>Bank transfer (automatic)</option>
                                <option>Credit card (automatic)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Phone Service</label>
                            <select id="PhoneService"><option>Yes</option><option>No</option></select>
                        </div>
                        <div class="form-group">
                            <label>Multiple Lines</label>
                            <select id="MultipleLines"><option>No</option><option>Yes</option><option>No phone service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Online Security</label>
                            <select id="OnlineSecurity"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Tech Support</label>
                            <select id="TechSupport"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Online Backup</label>
                            <select id="OnlineBackup"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Device Protection</label>
                            <select id="DeviceProtection"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Streaming TV</label>
                            <select id="StreamingTV"><option>Yes</option><option>No</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Streaming Movies</label>
                            <select id="StreamingMovies"><option>No</option><option>Yes</option><option>No internet service</option></select>
                        </div>
                        <div class="form-group">
                            <label>Paperless Billing</label>
                            <select id="PaperlessBilling"><option>Yes</option><option>No</option></select>
                        </div>
                        <div class="form-group">
                            <label>Monthly Charges ($)</label>
                            <input type="number" step="0.05" id="MonthlyCharges" value="79.85">
                        </div>
                        <div class="form-group">
                            <label>Total Charges ($)</label>
                            <input type="number" step="0.05" id="TotalCharges" value="159.70">
                        </div>
                    </div>

                    <div class="btn-group">
                        <button type="button" class="btn-primary" onclick="predictChurn()">⚡ Run Churn Prediction</button>
                        <button type="button" class="btn-preset" onclick="fillSample('high')">Load High-Risk Sample</button>
                        <button type="button" class="btn-preset" onclick="fillSample('low')">Load Low-Risk Sample</button>
                    </div>
                </form>
            </div>

            <!-- Right: Results & Monitoring -->
            <div>
                <div class="card" style="margin-bottom: 20px;">
                    <h2>Prediction Result</h2>
                    <div id="placeholder" style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 30px 10px;">
                        Adjust the customer profile and click <strong>Run Churn Prediction</strong> to test in real-time.
                    </div>
                    <div id="resultBox" class="result-box">
                        <div id="badgeRisk" class="badge"></div>
                        <div id="resTitle" class="result-title"></div>
                        <div id="resProb" class="result-prob"></div>
                        <div id="resDetail" style="font-size: 13px; color: var(--text-muted);"></div>
                        <pre id="jsonOutput"></pre>
                    </div>
                </div>

                <div class="card metrics-card">
                    <h2>Production Model Telemetry</h2>
                    <div class="metric-row"><span class="metric-label">Model Type</span><span class="metric-val">Linear SVM (C=0.1)</span></div>
                    <div class="metric-row"><span class="metric-label">ROC-AUC Score</span><span class="metric-val">0.8262</span></div>
                    <div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-val">78.85%</span></div>
                    <div class="metric-row"><span class="metric-label">F1-Score</span><span class="metric-val">57.06%</span></div>
                    <div class="metric-row"><span class="metric-label">Registry Stage</span><span class="metric-val" style="color: #4ade80;">Production (v2.0)</span></div>
                    <div class="metric-row"><span class="metric-label">Validation Engine</span><span class="metric-val">Pydantic v2</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function fillSample(type) {
            if(type === 'high') {
                document.getElementById('Contract').value = 'Month-to-month';
                document.getElementById('tenure').value = 2;
                document.getElementById('InternetService').value = 'Fiber optic';
                document.getElementById('PaymentMethod').value = 'Electronic check';
                document.getElementById('OnlineSecurity').value = 'No';
                document.getElementById('TechSupport').value = 'No';
                document.getElementById('MonthlyCharges').value = 85.0;
                document.getElementById('TotalCharges').value = 170.0;
            } else {
                document.getElementById('Contract').value = 'Two year';
                document.getElementById('tenure').value = 60;
                document.getElementById('InternetService').value = 'DSL';
                document.getElementById('PaymentMethod').value = 'Credit card (automatic)';
                document.getElementById('OnlineSecurity').value = 'Yes';
                document.getElementById('TechSupport').value = 'Yes';
                document.getElementById('MonthlyCharges').value = 45.0;
                document.getElementById('TotalCharges').value = 2700.0;
            }
            predictChurn();
        }

        async function predictChurn() {
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
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                document.getElementById('placeholder').style.display = 'none';
                const box = document.getElementById('resultBox');
                box.style.display = 'block';

                const probPercent = (data.churn_probability * 100).toFixed(1) + '%';
                document.getElementById('resProb').innerText = probPercent;
                document.getElementById('jsonOutput').innerText = JSON.stringify(data, null, 2);

                if (data.churn_prediction === 1) {
                    box.className = 'result-box result-high';
                    document.getElementById('resTitle').innerText = 'Customer Will CHURN';
                    document.getElementById('resTitle').style.color = '#ef4444';
                    document.getElementById('badgeRisk').innerText = 'HIGH CHURN RISK';
                    document.getElementById('badgeRisk').className = 'badge badge-danger';
                    document.getElementById('resDetail').innerText = 'High risk detected. Recommend retention discount.';
                } else {
                    box.className = 'result-box result-low';
                    document.getElementById('resTitle').innerText = 'Customer Will RETAIN';
                    document.getElementById('resTitle').style.color = '#10b981';
                    document.getElementById('badgeRisk').innerText = data.risk_level.toUpperCase() + ' RISK';
                    document.getElementById('badgeRisk').className = 'badge badge-success';
                    document.getElementById('resDetail').innerText = 'Customer is satisfied and unlikely to leave.';
                }
            } catch (err) {
                alert('Inference error: ' + err);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

