# Santander Customer Transaction Prediction

My attempt at the [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) Kaggle competition.

Binary classification on 200 anonymised numerical features to predict whether a customer will make a specific transaction. ~10% positive class, evaluated on AUC-ROC.

## Approach

### Part 1 — Gaussian Naive Bayes
- Row-wise statistical aggregates (mean, std, min, max, sum, skew, kurtosis) added as extra features
- `var_smoothing` grid-searched via stratified 5-fold CV
- NB v2: added per-feature value-count features — degraded due to overconfident predictions
- NB v3: applied `QuantileTransformer` to enforce Gaussianity → best NB result

### Part 2 — LightGBM
- Detected and filtered ~100k synthetic rows in the test set using unique-value signatures
- Value-count features computed on real data only (train + real test) to avoid synthetic noise
- 5-fold stratified OOF training with early stopping
- Hyperparameter tuning via Optuna (75 trials, MedianPruner, 30% subsample for speed)

## Results

| Model | CV AUC | Kaggle Public | Kaggle Private |
|-------|--------|---------------|----------------|
| GaussianNB v1 (row-wise features) | 0.8836 | 0.88392 | 0.88244 |
| GaussianNB v3 (QuantileTransformer) | 0.8836 | 0.88579 | 0.88367 |
| LightGBM (Optuna-tuned, 5-fold OOF) | 0.9073 | **0.91034** | **0.90730** |

## Stack
Python · scikit-learn · LightGBM · Optuna
