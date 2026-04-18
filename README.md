# Boston Housing Price Prediction

Course assignment · Machine Learning with scikit-learn

## Overview

End-to-end regression pipeline on the Boston Housing dataset, following the modeling workflow from Géron's *Hands-On Machine Learning* (Chapter 2). Covers EDA, feature engineering, sklearn pipeline construction, cross-validated model comparison, and hyperparameter tuning.

## Workflow

- EDA — correlation analysis, distribution plots, scatter plots for top features
- Feature engineering — 4 derived features (`ROOM_LSTAT`, `LOG_LSTAT`, `LOG_CRIM`, `TAX_CRIME`)
- sklearn `Pipeline` with `ColumnTransformer` + `StandardScaler`
- 5-fold CV comparison across `LinearRegression`, `DecisionTree`, `RandomForest`, `SVR`
- `GridSearchCV` tuning for RandomForest and SVR
- Feature importance analysis via tuned RandomForest

## Results

| Model | CV RMSE | Test RMSE |
|---|---|---|
| LinearRegression | 4.38 | 3.89 |
| DecisionTree | 4.43 | 3.97 |
| RandomForest (tuned) | 3.31 | 3.27 |
| **SVR (tuned)** | **3.25** | **2.83** ✓ best |

SVR with RBF kernel (C=100, ε=0.5) achieves the best test RMSE of **2.83** ($2,830 avg error). The engineered feature `ROOM_LSTAT` (RM / LSTAT) ranks #1 in RF importance at 0.35, above all original features.

## Stack

- `pandas`, `numpy`, `matplotlib`
- `scikit-learn` — `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `cross_val_score`

## Usage

```bash
git clone https://github.com/<your-username>/boston-housing-pipeline
jupyter notebook boston_pipeline_fixed.ipynb
```

## Reference

Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Chapter 2.
