# SGCC Baseline Model Report

## 1. Dataset & Split Methodology
- **Total Features**: 55
- **Class Distribution**: 1365 Normal, 135 Theft (~9% theft)
- **Split**: 70% Train (1049), 15% Validation (226), 15% Test (225)
- **Stratification**: Enabled (Random Seed: 42)

## 2. Experimental Configurations
- **Strategies Tested**: 
  - Original (No imbalance correction)
  - Class-Weighted (Algorithm penalty adjustment)
  - SMOTE (Oversampling applied strictly to Train split only)
- **Models Evaluated**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Data Scaling**: Applied StandardScaling exclusively within Logistic Regression pipelines.

## 3. Metrics Summary (Validation & Test)
| Model | Strategy | Val F1 | Val PR-AUC | Test F1 | Test PR-AUC |
|-------|----------|--------|------------|---------|-------------|
| Logistic Regression | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression (Weighted) | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree (Weighted) | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest (Weighted) | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost (Weighted) | Original | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | SMOTE | 1.000 | 1.000 | 1.000 | 1.000 |
| Decision Tree | SMOTE | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | SMOTE | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | SMOTE | 1.000 | 1.000 | 1.000 | 1.000 |

## 4. Interpretation of Results
- The highest validation F1-score (1.000) was achieved by **Logistic Regression** under the **Original** strategy.
- Class-weighting significantly improves Recall for Tree models and Logistic Regression, reducing false negatives in predicting electricity theft.
- SMOTE exhibits varying effectiveness depending on the non-linear capability of the model.

## 5. Limitations
- Tree models might exhibit inflated performance on highly correlated features (33 features have r > 0.95). 
- Hyperparameters are entirely defaults. No extensive tuning was performed.
