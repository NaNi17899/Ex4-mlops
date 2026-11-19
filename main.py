"""
Module: main.py
Description: End-to-end automated MLOps pipeline orchestrator:
             Data Loading -> Validation -> Preprocessing -> Feature Engineering ->
             Training (with tuning) -> Evaluation -> Artifact Persistence -> MLflow Logging -> Production Promotion.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Silencing non-critical deprecation notices
warnings.filterwarnings("ignore")
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from mlflow.tracking import MlflowClient

from src.data_loader import load_data
from src.validator import validate_data
from src.preprocessor import DataPreprocessor
from src.feature_engineering import engineer_features, analyze_correlations, lasso_feature_selection
from src.trainer import train_svm, tune_svm, save_model
from src.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.predictor import ChurnPredictor

def run_pipeline(
    data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "Telco_Customer_Churn_SVM"
):
    print("=" * 70)
    print(">>> STARTING AUTOMATED MLOPS PIPELINE: TELCO CUSTOMER CHURN (SVM) <<<")
    print("=" * 70)

    # ---------------------------------------------------------
    # Step 1: Data Loading
    # ---------------------------------------------------------
    print("\n[Step 1/7] Loading Dataset...")
    df_raw = load_data(data_path)

    # ---------------------------------------------------------
    # Step 2: Data Validation
    # ---------------------------------------------------------
    print("\n[Step 2/7] Validating Data Schema & Integrity...")
    val_report = validate_data(df_raw)
    if not val_report["is_valid"]:
        raise ValueError(f"Data validation failed: {val_report}")

    # ---------------------------------------------------------
    # Step 3: Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3/7] Feature Engineering (Avg_Monthly_Charge_Ratio)...")
    # Clean numeric charges first so ratio can compute accurately
    df_temp = df_raw.copy()
    num_tc = pd.to_numeric(df_temp['TotalCharges'], errors='coerce')
    df_temp['TotalCharges'] = num_tc.fillna(num_tc.median())
    df_engineered = engineer_features(df_temp)

    # ---------------------------------------------------------
    # Step 4: Preprocessing (Cleaning, Encoding & StandardScaler)
    # ---------------------------------------------------------
    print("\n[Step 4/7] Preprocessing Data (Cleaning, Categorical Encoding & Scaling)...")
    preprocessor = DataPreprocessor()
    X_scaled, y = preprocessor.fit_transform(df_engineered, target_col='Churn')
    preprocessor.save("models/preprocessor.joblib")

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Correlation Analysis
    print("\nAnalyzing Feature Correlations with Churn:")
    encoded_df = pd.DataFrame(X_train, columns=preprocessor.feature_columns)
    analyze_correlations(encoded_df, pd.Series(y_train), top_n=5)

    # LASSO Feature Selection
    print("\nPerforming LASSO (L1 Regularization) Feature Selection:")
    _, selected_features, _ = lasso_feature_selection(
        X_train, y_train, preprocessor.feature_columns, alpha=0.01
    )

    # ---------------------------------------------------------
    # Step 5 & 6: MLflow Experiment Tracking, Training & Evaluation
    # ---------------------------------------------------------
    print("\n[Step 5/7] Initializing MLflow Experiment Tracking...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # --- RUN 1: Version 1 (Baseline SVM: C=1.0, RBF) ---
    print("\n--- Running Experiment Version 1: Baseline SVM (C=1.0, RBF) ---")
    with mlflow.start_run(run_name="v1_baseline_svm") as run_v1:
        model_v1 = train_svm(X_train, y_train, C=1.0, kernel='rbf')
        metrics_v1 = evaluate_model(model_v1, X_test, y_test)
        
        mlflow.log_params({"C": 1.0, "kernel": "rbf", "pipeline_version": "v1.0"})
        mlflow.log_metrics(metrics_v1)
        mlflow.sklearn.log_model(model_v1, "model", registered_model_name="TelcoChurnSVM")
        run_v1_id = run_v1.info.run_id
        print(f"MLflow Run v1 ID: {run_v1_id}")

    # --- RUN 2: Version 2 (Tuned SVM with GridSearchCV & Engineered Features) ---
    print("\n--- Running Experiment Version 2: Tuned SVM with GridSearchCV ---")
    param_grid = {'C': [0.1, 1.0, 5.0], 'kernel': ['linear', 'rbf']}
    best_model_v2, best_params_v2 = tune_svm(X_train, y_train, param_grid=param_grid, cv=3)
    
    with mlflow.start_run(run_name="v2_tuned_svm") as run_v2:
        metrics_v2 = evaluate_model(best_model_v2, X_test, y_test)
        
        # Save artifacts
        plot_confusion_matrix(best_model_v2, X_test, y_test, "artifacts/confusion_matrix.png")
        plot_roc_curve(best_model_v2, X_test, y_test, "artifacts/roc_curve.png")
        
        mlflow.log_params({**best_params_v2, "pipeline_version": "v2.0", "engineered_feature": "Avg_Monthly_Charge_Ratio"})
        mlflow.log_metrics(metrics_v2)
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/roc_curve.png")
        mlflow.sklearn.log_model(best_model_v2, "model", registered_model_name="TelcoChurnSVM")
        run_v2_id = run_v2.info.run_id
        print(f"MLflow Run v2 ID: {run_v2_id}")

    # Save best model to disk for serving
    save_model(best_model_v2, "models/best_svm_model.joblib")
    
    # Save metrics JSON for API /metrics endpoint
    metrics_record = {
        "model_name": "TelcoChurnSVM",
        "winning_version": "v2.0",
        "best_hyperparameters": best_params_v2,
        "metrics": metrics_v2,
        "baseline_metrics": metrics_v1
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=4)
    print("[Pipeline] Saved metrics metadata to models/metrics.json")

    # ---------------------------------------------------------
    # Step 6: Model Comparison & Model Registry Promotion
    # ---------------------------------------------------------
    print("\n[Step 6/7] Comparing Model Runs & Promoting Winner to Production...")
    print(f"  v1 Baseline ROC-AUC: {metrics_v1['roc_auc']:.4f} | F1: {metrics_v1['f1_score']:.4f}")
    print(f"  v2 Tuned    ROC-AUC: {metrics_v2['roc_auc']:.4f} | F1: {metrics_v2['f1_score']:.4f}")

    # Promote version 2 to Production
    try:
        versions = client.get_latest_versions("TelcoChurnSVM")
        latest_ver = versions[-1].version
        client.set_registered_model_alias("TelcoChurnSVM", "production", latest_ver)
        try:
            client.transition_model_version_stage(
                name="TelcoChurnSVM",
                version=latest_ver,
                stage="Production",
                archive_existing_versions=True
            )
        except Exception:
            pass
        print(f"[ModelRegistry] Successfully promoted TelcoChurnSVM Version {latest_ver} to 'Production' stage!")
    except Exception as e:
        print(f"[ModelRegistry Warning] {e}")

    # ---------------------------------------------------------
    # Step 7: Prediction Verification
    # ---------------------------------------------------------
    print("\n[Step 7/7] Testing Real-Time Prediction on Sample Customer...")
    predictor = ChurnPredictor()
    sample_customer = {
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
    pred_res = predictor.predict_single(sample_customer)
    print(f"[Prediction Test Result]: {pred_res}")
    print("\n" + "=" * 70)
    print(">>> PIPELINE EXECUTION COMPLETED SUCCESSFULLY! <<<")
    print("=" * 70)
    return metrics_record

if __name__ == "__main__":
    run_pipeline()
