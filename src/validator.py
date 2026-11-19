"""
Module: validator.py
Description: Validates data schema, column types, integrity, and constraints.
"""

import pandas as pd
from typing import Dict, Any

REQUIRED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates data against predefined schema and constraints.
    """
    report = {
        "is_valid": True,
        "missing_columns": [],
        "duplicate_rows": 0,
        "blank_total_charges": 0,
        "unique_churn_values": []
    }
    
    # Check missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["missing_columns"] = missing_cols
        print(f"[Validator Error] Missing columns: {missing_cols}")
    
    # Check duplicates by customerID if present
    if 'customerID' in df.columns:
        report["duplicate_rows"] = int(df.duplicated(subset=['customerID']).sum())
    
    # Check for blank strings in TotalCharges
    if 'TotalCharges' in df.columns:
        # blanks or spaces
        blank_count = df['TotalCharges'].astype(str).str.strip().eq('').sum()
        report["blank_total_charges"] = int(blank_count)
    
    # Check target distribution
    if 'Churn' in df.columns:
        report["unique_churn_values"] = df['Churn'].unique().tolist()
        if not set(df['Churn'].dropna().unique()).issubset({'Yes', 'No', 1, 0}):
            report["is_valid"] = False
            print("[Validator Warning] Churn column contains unexpected values.")

    print(f"[Validator] Validation Report: Valid={report['is_valid']}, "
          f"Duplicates={report['duplicate_rows']}, "
          f"Blank TotalCharges={report['blank_total_charges']}")
    
    return report

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    rep = validate_data(df)
    print(rep)
