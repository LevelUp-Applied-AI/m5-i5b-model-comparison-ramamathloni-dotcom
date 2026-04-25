"""
Module 5 Week B — Integration Task: Model Comparison & Decision Memo
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, average_precision_score, 
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay

# Fixed numeric features list
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]

def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42):
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES]
    y = df['churned']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

def define_models():
    return {
        'Dummy': Pipeline([('scaler', 'passthrough'), ('model', DummyClassifier(strategy='most_frequent'))]),
        'LR_default': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))]),
        'LR_balanced': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
        'DT_depth5': Pipeline([('scaler', 'passthrough'), ('model', DecisionTreeClassifier(max_depth=5, random_state=42))]),
        'RF_default': Pipeline([('scaler', 'passthrough'), ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))]),
        'RF_balanced': Pipeline([('scaler', 'passthrough'), ('model', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42))]),
    }

def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    
    for name, pipeline in models.items():
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'pr_auc': []}
        
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            metrics['pr_auc'].append(average_precision_score(y_val, y_prob))
        
        row = {'model': name}
        for m, vals in metrics.items():
            row[f'{m}_mean'] = np.mean(vals)
            row[f'{m}_std'] = np.std(vals)
        results.append(row)
        
    return pd.DataFrame(results)

def save_comparison_table(results_df, output_path="results/comparison_table.csv"):
    results_df.to_csv(output_path, index=False)

def plot_pr_curves_top3(models, X_test, y_test, output_path="results/pr_curves.png"):
    scores = [(name, average_precision_score(y_test, m.predict_proba(X_test)[:, 1])) for name, m in models.items()]
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _ in top3:
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)
    plt.title("PR Curves (Top 3)")
    plt.savefig(output_path)
    plt.close()

def plot_calibration_top3(models, X_test, y_test, output_path="results/calibration.png"):
    scores = [(name, average_precision_score(y_test, m.predict_proba(X_test)[:, 1])) for name, m in models.items()]
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, _ in top3:
        CalibrationDisplay.from_estimator(models[name], X_test, y_test, n_bins=10, ax=ax, name=name)
    plt.title("Calibration Curves (Top 3)")
    plt.savefig(output_path)
    plt.close()

def save_best_model(best_model, output_path="results/best_model.joblib"):
    dump(best_model, output_path)

def log_experiment(results_df, output_path="results/experiment_log.csv"):
    log_df = results_df[['model', 'accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'pr_auc_mean']].copy()
    log_df.columns = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'pr_auc']
    log_df['timestamp'] = datetime.now().isoformat()
    log_df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))

def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test, y_test, feature_names, min_diff=0.15):
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    diff = np.abs(rf_prob - lr_prob)
    idx = np.argmax(diff)
    if diff[idx] < min_diff: return None
    return {
        'sample_idx': idx,
        'feature_values': X_test.iloc[idx].to_dict(),
        'rf_proba': rf_prob[idx],
        'lr_proba': lr_prob[idx],
        'prob_diff': diff[idx],
        'true_label': y_test.iloc[idx]
    }

def main():
    print("Starting integration tasks...")
    os.makedirs("results", exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_and_preprocess()
    models = define_models()
    results_df = run_cv_comparison(models, X_train, y_train)
    save_comparison_table(results_df)
    
    fitted_models = {name: m.fit(X_train, y_train) for name, m in models.items()}
    plot_pr_curves_top3(fitted_models, X_test, y_test)
    plot_calibration_top3(fitted_models, X_test, y_test)
    
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    save_best_model(fitted_models[best_name])
    log_experiment(results_df)
    
    disagreement = find_tree_vs_linear_disagreement(fitted_models["RF_default"], fitted_models["LR_default"], X_test, y_test, NUMERIC_FEATURES)
    if disagreement:
        with open("results/tree_vs_linear_disagreement.md", "w") as f:
            f.write("# Tree vs. Linear Disagreement Analysis\n\n")
            f.write(f"- **Sample Index:** {disagreement['sample_idx']}\n")
            f.write(f"- **True Label:** {disagreement['true_label']}\n")
            f.write(f"- **RF Proba:** {disagreement['rf_proba']:.4f}\n")
            f.write(f"- **LR Proba:** {disagreement['lr_proba']:.4f}\n")
            f.write("\n## Structural Explanation\n[Fill in your analysis here]\n")
    
    print("\n--- All tasks completed successfully! ---")

if __name__ == "__main__":
    main()