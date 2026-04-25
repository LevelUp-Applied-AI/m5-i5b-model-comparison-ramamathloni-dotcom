# tier3_framework.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_comparison import load_and_preprocess, define_models

class ModelSelector:
    """
    A framework to manage model training, evaluation, and selection.
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = define_models()
        self.results = []

    def run_all(self):
        """Train and evaluate all models defined in define_models()."""
        print("Starting comprehensive model evaluation...")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            self.results.append({
                'model': name,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1': f1_score(self.y_test, y_pred, zero_division=0)
            })
        print("Evaluation complete.")

    def get_best_model(self, metric='f1'):
        """Returns the name of the best model based on a chosen metric."""
        df = pd.DataFrame(self.results)
        best_row = df.loc[df[metric].idxmax()]
        return best_row['model'], best_row[metric]

    def summary(self):
        """Prints a clean summary of all model performances."""
        return pd.DataFrame(self.results).set_index('model')

if __name__ == "__main__":
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # 2. Initialize the Framework
    selector = ModelSelector(X_train, X_test, y_train, y_test)
    
    # 3. Run evaluation
    selector.run_all()
    
    # 4. Display results
    print("\nModel Performance Summary:")
    print(selector.summary())
    
    # 5. Get winner
    best_name, score = selector.get_best_model(metric='f1')
    print(f"\nWinner based on F1 Score: {best_name} (Score: {score:.4f})")