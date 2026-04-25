# Tree vs. Linear Disagreement Analysis

- **Sample Index:** 777
- **True Label:** 0
- **RF Proba:** 0.5998
- **LR Proba:** 0.1700

## Structural Explanation
# Tree vs. Linear Disagreement Analysis

## Structural Explanation
The disagreement between the Random Forest (RF) and Logistic Regression (LR) models typically arises from the fundamental difference in their decision logic:

1. **Handling of Non-Linearity:** The RF model captures non-linear relationships and interactions between features (e.g., the complex interaction between `tenure` and `contract_months`) by partitioning the feature space into hyper-rectangles. In contrast, the LR model assumes a linear combination of inputs. If a relationship is highly non-linear, the LR model may fail to capture it, leading to a different probability estimate.
2. **Feature Interaction:** Trees effectively learn feature interactions without explicit specification. If a specific sample has a unique combination of `monthly_charges` and `num_support_calls` that pushes it towards a specific threshold, the RF model will isolate this, whereas the LR model will weight these features independently, leading to divergent predictions.

In this instance, the RF model identified the sample as having a higher probability of churn due to the specific feature interactions present in the sample data, which the linear decision boundary of the LR model likely smoothed over.
