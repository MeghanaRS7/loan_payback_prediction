# 🏆 Kaggle - Loan Repayment Prediction

This repository contains the solution and experiments for the S5E11 Kaggle competition.  
The goal is to predict whether a borrower will repay a loan using tabular data.

| Metric     | Score                     |
| ---------- | ------------------------- |
| Public AUC | **0.92724**               |
| Rank       | **~183 / 9,703 entrants** |



## 🔬 Approach

The current pipeline uses:
- 10-fold target encoding (smoothed)
- frequency encoding + quantile binning
- domain features like rounded income / loan buckets, grade–subgrade split, and debt burden
- LightGBM 7-fold CV to find best iteration, then +300 rounds
- 5-seed ensemble for final predictions

Optional (code included but not used in final submission):
- XGBoost baseline
- linear blending between LightGBM and XGBoost

---

## ▶️ Run the code

Place `train.csv` and `test.csv` in the repository root, then run:

```bash
python train_lgbm.py

