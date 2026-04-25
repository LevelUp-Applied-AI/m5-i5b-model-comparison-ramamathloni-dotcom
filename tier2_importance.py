# tier2_importance.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.inspection import permutation_importance
from model_comparison import load_and_preprocess, define_models

def calculate_and_plot_importance(model, X_test, y_test, feature_names):
    """
    Calculates permutation importance and plots it as a horizontal bar chart.
    """
    # Calculate permutation importance
    # n_repeats=10 means we shuffle each feature 10 times to measure impact
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Store results in a DataFrame for easy plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance_mean'], xerr=importance_df['importance_std'])
    plt.xlabel('Decrease in Model Accuracy (Importance)')
    plt.title('Permutation Feature Importance (Best Model)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/feature_importance.png")
    print("Feature importance plot saved to results/feature_importance.png.")
    
    return importance_df

if __name__ == "__main__":
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    models = define_models()
    
    # 2. Use the Best Model (e.g., RF_default)
    best_model = models['RF_default']
    best_model.fit(X_train, y_train)
    
    # 3. Calculate and plot
    feature_names = X_train.columns
    importance_df = calculate_and_plot_importance(best_model, X_test, y_test, feature_names)
    
    # 4. Display ranking
    print("\nFeature Importance Ranking:")
    print(importance_df[['feature', 'importance_mean']])