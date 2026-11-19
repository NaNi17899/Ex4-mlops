# Automated MLOps Pipeline for Telco Customer Churn Prediction
### **Algorithm: Support Vector Machine (SVM) | Dataset: Telco Customer Churn**

An end-to-end, production-grade MLOps engineering system implementing **Experiment 4 (SET-4)** on the Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).

---

## 📋 Table of Contents
- [Project Architecture](#-project-architecture)
- [Experiment 4 Sub-questions Mapping (a - f)](#-experiment-4-sub-questions-mapping-a---f)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Jupyter Notebook](#-running-the-jupyter-notebook)
- [Automated Pipeline Execution (`main.py`)](#-automated-pipeline-execution-mainpy)
- [FastAPI Real-Time Prediction Serving](#-fastapi-real-time-prediction-serving)
- [Dataset Versioning with DVC & Git](#-dataset-versioning-with-dvc--git)
- [MLflow Experiment Tracking & Model Registry](#-mlflow-experiment-tracking--model-registry)
- [Verification & Test Results](#-verification--test-results)

---

## 🏛️ Project Architecture

```
[Raw Dataset] (WA_Fn-UseC_-Telco-Customer-Churn.csv)
      │
      ▼
[src/data_loader.py] ──> [src/validator.py] (Schema & Constraint Validation)
      │
      ▼
[src/feature_engineering.py] (Avg_Monthly_Charge_Ratio, Correlation, LASSO)
      │
      ▼
[src/preprocessor.py] (Median Imputation, Deduplication, One-Hot Encoding, StandardScaler)
      │
      ├──> DVC Remote Storage Tracking (v1.0 & v2.0)
      │
      ▼
[src/trainer.py] (SVM Training & GridSearchCV Hyperparameter Tuning over C & kernel)
      │
      ▼
[src/evaluator.py] (Accuracy, Precision, Recall, F1-Score, ROC-AUC, CM & ROC Plots)
      │
      ├──> MLflow Experiment Tracking & Model Registry (sqlite:///mlflow.db)
      │    └──> Promotion to 'Production'
      │
      ▼
[models/] (best_svm_model.joblib, preprocessor.joblib, metrics.json)
      │
      ▼
[app.py] (FastAPI Microservice with Pydantic Input Validation)
      │
      ├──> GET  /metrics  (Production model validation metrics)
      ├──> POST /predict  (Real-time validated churn probability & risk classification)
      └──> GET  /docs     (Interactive OpenAPI documentation)
```

---

## 📑 Experiment 4 Sub-questions Mapping (a - f)

| Sub-question | Technical Task | Implementation Module | Output / Deliverable |
| :--- | :--- | :--- | :--- |
| **a)** | TotalCharges numeric coercion, median imputation, duplicate customer removal, categorical encoding, StandardScaler | `src/preprocessor.py` | `models/preprocessor.joblib`, zero missing/duplicate rows |
| **b)** | Feature Engineering (`Avg_Monthly_Charge_Ratio`), correlation analysis, LASSO selection, SVM tuning ($C$ & `kernel`), metric reporting | `src/feature_engineering.py`, `src/trainer.py`, `src/evaluator.py` | Best model: $C=0.1$, Linear; ROC-AUC: **0.8262**, F1: **0.5706** |
| **c)** | Dataset & artifact tracking across $v1$ (5k samples) and $v2$ (7k samples) using Git & DVC remote storage | `src/dvc_versioning_demo.py`, `.dvc/` | Reproducible version switching demonstrated via `dvc checkout` |
| **d)** | MLflow parameter/metric logging, serialized model logging, runs comparison, promotion to Model Registry `Production` stage | `main.py`, MLflow Client | `sqlite:///mlflow.db`, Registered model: `TelcoChurnSVM` (Production) |
| **e)** | End-to-end automation of pipeline via modular Python scripts orchestrated by `main.py` | `src/` modules + `main.py` | Seamless execution from raw data to saved model & inference |
| **f)** | Real-time churn prediction microservice with Pydantic input validation, testing `/predict` & `/metrics` with valid & invalid payloads | `app.py`, `test_api.py` | FastAPI server, 422 rejection tests, automated test suite |

---

## 📂 Project Structure

```
mlops-external/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Kaggle Telco Churn raw dataset
│   ├── dataset.csv                               # Active DVC-tracked dataset
│   └── dataset.csv.dvc                           # DVC metadata tracking hash
├── src/
│   ├── __init__.py
│   ├── data_loader.py                            # Dataset ingestion
│   ├── validator.py                              # Data schema & constraint validator
│   ├── preprocessor.py                           # Cleaning, encoding & StandardScaler
│   ├── feature_engineering.py                    # Avg_Monthly_Charge_Ratio & LASSO selection
│   ├── trainer.py                                # SVM training & GridSearchCV tuning
│   ├── evaluator.py                              # Evaluation metrics & diagnostic plotting
│   ├── predictor.py                              # Inference engine
│   └── dvc_versioning_demo.py                    # DVC v1/v2 tracking & switching demo
├── models/
│   ├── best_svm_model.joblib                     # Serialized production SVM model
│   ├── preprocessor.joblib                       # Fitted scaler & feature encoder
│   └── metrics.json                              # Production metrics metadata
├── artifacts/
│   ├── confusion_matrix.png                      # Confusion matrix plot
│   └── roc_curve.png                             # Receiver Operating Characteristic plot
├── Experiment_4_Telco_Churn_MLOps.ipynb          # Master interactive Jupyter Notebook
├── main.py                                       # Pipeline orchestrator script
├── app.py                                        # FastAPI serving application
├── test_api.py                                   # Automated API endpoint test suite
├── build_notebook.py                             # Script to compile & execute notebook
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone repository and install dependencies:
```bash
git clone https://github.com/NaNi17899/mlops-telco-churn-svm.git
cd mlops-telco-churn-svm
pip install -r requirements.txt
```

### 2. Running in Jupyter Notebook
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Open **`Experiment_4_Telco_Churn_MLOps.ipynb`**. The entire notebook has pre-computed live outputs, markdown explanations, interactive visualizations, and test suite results.

### 3. Running the Automated Pipeline via CLI
Execute the complete end-to-end MLOps pipeline:
```bash
python main.py
```

### 4. Running the FastAPI Server
Start the production API server:
```bash
python app.py
```
Or via Uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- Redoc UI: `http://localhost:8000/redoc`

### 5. Running the API Test Suite
Run automated unit and schema validation tests (testing valid & invalid payloads):
```bash
python test_api.py
```

---

## 🔍 DVC Version Switching & Reproducibility
To switch between versions:

```bash
# Switch to Version 1 (5,000 samples)
git checkout v1.0 -- data/dataset.csv.dvc
python -m dvc checkout data/dataset.csv.dvc

# Switch to Version 2 (Full 7,043 samples)
git checkout v2.0 -- data/dataset.csv.dvc
python -m dvc checkout data/dataset.csv.dvc
```

---

## 📈 MLflow Experiment UI
To inspect tracked experiments, parameter runs, metrics, and Model Registry in the web dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Navigate to `http://localhost:5000`.

---

## 📊 Model Evaluation Results

| Metric | Version 1 (Baseline: C=1.0, RBF) | Version 2 (Tuned: C=0.1, Linear) |
| :--- | :---: | :---: |
| **Accuracy** | 79.21% | 78.85% |
| **Precision** | 64.11% | 61.88% |
| **Recall** | 49.20% | **52.94%** |
| **F1-Score** | 55.67% | **57.06%** |
| **ROC-AUC** | 0.7970 | **0.8262** |
| **Registry Stage** | Archived | **Production** |
