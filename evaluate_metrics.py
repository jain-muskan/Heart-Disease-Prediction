# -*- coding: utf-8 -*-
"""
Evaluation and Metrics Comparison Script
Compares standard baseline classifiers against the research-adapted 
multivariable distance & iterative sampling architecture.
Repository: https://github.com/jain-muskan/Heart-Disease-Prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(y_true, y_pred, model_name):
    """
    Computes core classification metrics for a given model.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4)
    }

def generate_comparison_report(results_list):
    """
    Generates a clean tabular summary comparing all model metrics.
    """
    df_metrics = pd.DataFrame(results_list)
    print("=" * 70)
    print(" PERFORMANCE COMPARISON REPORT: BASELINE VS. RESEARCH ARCHITECTURE")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70)
    return df_metrics

# Example execution stub (assuming y_test and predictions from heart_disease_pred.py exist):
if __name__ == "__main__":
    # Mock evaluation metrics dictionary for demonstration purposes
    sample_results = [
        {"Model": "Logistic Regression (Baseline)", "Accuracy": 0.6725, "Precision": 0.4120, "Recall": 0.7050, "F1-Score": 0.5201},
        {"Model": "Random Forest Classifier", "Accuracy": 0.8410, "Precision": 0.7890, "Recall": 0.6540, "F1-Score": 0.7152},
        {"Model": "LightGBM Classifier", "Accuracy": 0.8520, "Precision": 0.8010, "Recall": 0.6820, "F1-Score": 0.7366},
        {"Model": "Research-Adapted Cluster Predictor (CDM + Iterative)", "Accuracy": 0.8650, "Precision": 0.8240, "Recall": 0.7120, "F1-Score": 0.7639}
    ]
    
    report_df = generate_comparison_report(sample_results)
