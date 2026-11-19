"""
Module: feature_engineering.py
Description: Feature engineering (Avg_Monthly_Charge_Ratio), correlation-based analysis,
             and LASSO (L1-penalized) feature selection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features:
    Avg_Monthly_Charge_Ratio = TotalCharges / (tenure * MonthlyCharges + epsilon)
    """
    df_feat = df.copy()
    
    # Calculate Avg_Monthly_Charge_Ratio
    # When tenure is 0, default ratio is 1.0 (baseline expected billing)
    tenure_charges = df_feat['tenure'] * df_feat['MonthlyCharges']
    ratio = df_feat['TotalCharges'] / (tenure_charges + 1e-5)
    
    # For customers with tenure == 0 or nulls, set to 1.0
    ratio = ratio.fillna(1.0).clip(lower=0.0, upper=5.0)
    df_feat['Avg_Monthly_Charge_Ratio'] = ratio
    
    print(f"[FeatureEngineering] Engineered feature 'Avg_Monthly_Charge_Ratio' added. Mean: {ratio.mean():.4f}")
    return df_feat

def analyze_correlations(X_df: pd.DataFrame, y: pd.Series, top_n: int = 10) -> pd.Series:
    """
    Computes Pearson correlation between features and binary target.
    """
    correlations = X_df.apply(lambda col: col.corr(y)).dropna()
    abs_corr = correlations.abs().sort_values(ascending=False)
    print(f"[FeatureEngineering] Top {top_n} features correlated with target:")
    for feat, corr_val in abs_corr.head(top_n).items():
        print(f"  - {feat}: {correlations[feat]:.4f} (abs: {corr_val:.4f})")
    return abs_corr

def lasso_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], alpha: float = 0.01) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Performs LASSO feature selection (L1 regularization) to select non-zero sparse features.
    """
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    
    selector = SelectFromModel(lasso, prefit=True)
    mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    X_selected = X[:, mask]
    
    print(f"[FeatureEngineering] LASSO (alpha={alpha}) selected {len(selected_features)} of {len(feature_names)} features.")
    return X_selected, selected_features, mask
