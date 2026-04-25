# tier1_threshold.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from model_comparison import load_and_preprocess, define_models

def sweep_thresholds(model, X_test, y_test):
    """
    Sweeps through thresholds from 0.1 to 0.9 and calculates 
    performance metrics at each step.
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    metrics = []
    
    # Get probability scores for the positive class (churn)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    for t in thresholds:
        # Convert probabilities to binary class based on current threshold
        y_pred = (y_prob >= t).astype(int)
        metrics.append({
            'threshold': t,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'num_alerts': np.sum(y_pred) # Number of customers targeted
        })
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    # 1. Load data and initialize models
    X_train, X_test, y_train, y_test = load_and_preprocess()
    models = define_models()
    
    # Select the model to analyze (e.g., Random Forest)
    best_model = models['RF_default']
    best_model.fit(X_train, y_train)
    
    # 2. Execute the sweep
    results = sweep_thresholds(best_model, X_test, y_test)
    
    # 3. Generate visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results['threshold'], results['precision'], label='Precision')
    plt.plot(results['threshold'], results['recall'], label='Recall')
    plt.plot(results['threshold'], results['f1'], label='F1 Score')
    
    # Add constraint line for business capacity (150 alerts per 10k customers)
    plt.axhline(y=150/10000, color='r', linestyle='--', label='Capacity Constraint (150/10k)')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score / Ratio')
    plt.title('Threshold Sweep Optimization')
    plt.legend()
    plt.grid(True)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/threshold_sweep.png")
    print("Threshold sweep plot saved to results/threshold_sweep.png.")
    
    # 4. Print detailed results to console for decision making
    print(results[['threshold', 'num_alerts', 'recall']])