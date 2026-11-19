"""
Module: evaluator.py
Description: Evaluates classifier performance, computes Accuracy, Precision, Recall,
             F1-Score, and ROC-AUC, and generates visual diagnostic plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluates model and returns key classification metrics dictionary.
    """
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    print("[Evaluator] Performance Metrics:")
    for k, v in metrics.items():
        print(f"  - {k.upper()}: {v:.4f}")
        
    return metrics

def plot_confusion_matrix(model: Any, X_test: np.ndarray, y_test: np.ndarray, save_path: str = "artifacts/confusion_matrix.png"):
    """
    Plots and saves Confusion Matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Telco Churn SVM - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluator] Confusion matrix saved to: {save_path}")

def plot_roc_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray, save_path: str = "artifacts/roc_curve.png"):
    """
    Plots and saves ROC Curve.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM ROC (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Evaluator] ROC curve saved to: {save_path}")
