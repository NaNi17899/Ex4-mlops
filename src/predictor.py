"""
Module: predictor.py
Description: Encapsulates end-to-end inference logic for new customer data.
"""

import os
import pandas as pd
from typing import Dict, Any, Union, List
from .preprocessor import DataPreprocessor
from .feature_engineering import engineer_features
from .trainer import load_model

class ChurnPredictor:
    def __init__(self, model_path: str = "models/best_svm_model.joblib", preprocessor_path: str = "models/preprocessor.joblib"):
        self.model = load_model(model_path)
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        print("[Predictor] Successfully initialized model and preprocessor.")

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes real-time inference for a single customer payload.
        """
        df_input = pd.DataFrame([customer_data])
        
        # Apply engineered feature
        df_engineered = engineer_features(df_input)
        
        # Transform through fitted preprocessor (encodes & scales)
        X_scaled = self.preprocessor.transform(df_engineered)
        
        # Inference
        pred_class = int(self.model.predict(X_scaled)[0])
        
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X_scaled)[0, 1])
        else:
            prob = float(1.0 / (1.0 + np.exp(-self.model.decision_function(X_scaled)[0])))
            
        result = {
            "churn_prediction": pred_class,
            "churn_label": "Yes" if pred_class == 1 else "No",
            "churn_probability": round(prob, 4),
            "risk_level": "High" if prob >= 0.6 else ("Medium" if prob >= 0.4 else "Low")
        }
        return result

    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes batch inference for multiple customer payloads.
        """
        return [self.predict_single(c) for c in customers]
