"""
Module: data_loader.py
Description: Handles loading the raw Telco Customer Churn dataset.
"""

import os
import pandas as pd
from typing import Tuple

def load_data(file_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """
    Loads dataset from the specified CSV file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[DataLoader] Successfully loaded dataset: {file_path}")
    print(f"[DataLoader] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head(2))
