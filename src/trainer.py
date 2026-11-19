"""
Module: trainer.py
Description: Trains Support Vector Machine (SVM) classifiers, supports hyperparameter tuning
             over C and kernel, and persists serialized model artifacts.
"""

import os
import joblib
import warnings
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Suppress scikit-learn 1.9 future warnings for probability parameter
warnings.filterwarnings("ignore", category=FutureWarning)

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = 'rbf',
    random_state: int = 42
) -> SVC:
    """
    Trains a baseline Support Vector Machine (SVM) classifier.
    """
    print(f"[Trainer] Training SVM with C={C}, kernel='{kernel}'...")
    model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    print("[Trainer] Training complete.")
    return model

def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any] = None,
    cv: int = 3,
    scoring: str = 'roc_auc'
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Performs Grid Search Cross-Validation to tune SVM hyperparameters (C and kernel).
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1.0, 5.0],
            'kernel': ['linear', 'rbf']
        }
        
    print(f"[Trainer] Hyperparameter tuning with GridSearchCV (scoring='{scoring}', cv={cv})...")
    base_svc = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(base_svc, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"[Trainer] Best Hyperparameters: {best_params} (CV {scoring}: {best_score:.4f})")
    return best_model, best_params

def save_model(model: Any, filepath: str = "models/best_svm_model.joblib"):
    """
    Serializes and persists model artifact to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[Trainer] Serialized model saved to: {filepath}")

def load_model(filepath: str = "models/best_svm_model.joblib") -> Any:
    """
    Loads serialized model artifact from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)
