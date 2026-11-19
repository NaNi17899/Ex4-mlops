"""
Module: preprocessor.py
Description: Data cleaning, missing value imputation, duplicate removal,
             categorical variable encoding, and numerical standardization (StandardScaler).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.total_charges_median: float = 0.0
        self.is_fitted: bool = False

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts TotalCharges to numeric, imputes missing values, and removes duplicates.
        """
        df_clean = df.copy()
        
        # Remove duplicate rows based on customerID if present
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['customerID'])
        
        # Coerce TotalCharges to numeric
        if 'TotalCharges' in df_clean.columns:
            tc_numeric = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            if not self.is_fitted:
                self.total_charges_median = float(tc_numeric.median() if not pd.isna(tc_numeric.median()) else 0.0)
            df_clean['TotalCharges'] = tc_numeric.fillna(self.total_charges_median)
        
        return df_clean

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits preprocessor on dataframe, encodes categorical variables, applies StandardScaler.
        Returns scaled feature matrix X and target vector y.
        """
        df_clean = self.clean_and_impute(df)
        
        # Extract target if present
        y = None
        if target_col in df_clean.columns:
            y = df_clean[target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).values
            df_features = df_clean.drop(columns=[target_col])
        else:
            df_features = df_clean
            
        if 'customerID' in df_features.columns:
            df_features = df_features.drop(columns=['customerID'])
            
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(df_features, drop_first=True)
        self.feature_columns = list(X_encoded.columns)
        
        # Fit and apply StandardScaler
        X_scaled = self.scaler.fit_transform(X_encoded)
        self.is_fitted = True
        
        print(f"[Preprocessor] Fit completed. Transformed features count: {len(self.feature_columns)}")
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new incoming data using the fitted columns and scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
            
        df_clean = self.clean_and_impute(df)
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])
        if 'Churn' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Churn'])
            
        X_encoded = pd.get_dummies(df_clean, drop_first=True)
        
        # Align columns with training feature columns (fill missing columns with 0)
        X_aligned = pd.DataFrame(0, index=X_encoded.index, columns=self.feature_columns)
        for col in X_encoded.columns:
            if col in self.feature_columns:
                X_aligned[col] = X_encoded[col]
                
        X_scaled = self.scaler.transform(X_aligned)
        return X_scaled

    def save(self, filepath: str = "models/preprocessor.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Preprocessor] Saved to: {filepath}")

    @staticmethod
    def load(filepath: str = "models/preprocessor.joblib") -> 'DataPreprocessor':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        return joblib.load(filepath)
