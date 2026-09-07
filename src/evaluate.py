"""
Model Evaluation and Benchmark Verification Module
Computes all classification metrics and generates statistical comparisons.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score,
    confusion_matrix, classification_report
)

def evaluate_model_performance(name: str, model, X_test, y_test) -> dict:
    """Computes comprehensive clinical prediction metrics."""
    preds = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = preds
        
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, probs),
        "MCC": matthews_corrcoef(y_test, preds),
        "Balanced Accuracy": balanced_accuracy_score(y_test, preds),
        "Cohen Kappa": cohen_kappa_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1 Score": f1_score(y_test, preds, zero_division=0)
    }

def get_paper_baseline_benchmarks() -> list:
    """Returns baseline and hyperparameter-tuned metrics directly from the 2025 paper."""
    return [
        {
            "Model": "Paper Best: Tuned Random Forest (Awe et al. 2025)",
            "Accuracy": 0.8195,
            "ROC AUC": 0.8696,
            "MCC": 0.6374,
            "Balanced Accuracy": 0.8187,
            "Cohen Kappa": 0.6374,
            "Precision": 0.8310,
            "Recall": 0.8310,
            "F1 Score": 0.8310
        },
        {
            "Model": "Paper: Gradient Boosting (Awe et al. 2025)",
            "Accuracy": 0.7293,
            "ROC AUC": 0.8092,
            "MCC": 0.4559,
            "Balanced Accuracy": 0.7230,
            "Cohen Kappa": 0.4505,
            "Precision": 0.7160,
            "Recall": 0.8169,
            "F1 Score": 0.7630
        },
        {
            "Model": "Paper: CatBoost (Awe et al. 2025)",
            "Accuracy": 0.7293,
            "ROC AUC": 0.7876,
            "MCC": 0.4551,
            "Balanced Accuracy": 0.7240,
            "Cohen Kappa": 0.4517,
            "Precision": 0.7215,
            "Recall": 0.8028,
            "F1 Score": 0.7600
        },
        {
            "Model": "Paper: XGBoost (Awe et al. 2025)",
            "Accuracy": 0.6917,
            "ROC AUC": 0.7703,
            "MCC": 0.3812,
            "Balanced Accuracy": 0.6826,
            "Cohen Kappa": 0.3710,
            "Precision": 0.6744,
            "Recall": 0.8169,
            "F1 Score": 0.7380
        },
        {
            "Model": "Paper: AdaBoost (Awe et al. 2025)",
            "Accuracy": 0.5789,
            "ROC AUC": 0.6336,
            "MCC": 0.1558,
            "Balanced Accuracy": 0.5780,
            "Cohen Kappa": 0.1557,
            "Precision": 0.6087,
            "Recall": 0.5915,
            "F1 Score": 0.6000
        }
    ]
