import pandas as pd
import numpy as np
import warnings
import pickle
import os
import lightgbm as lgb
import xgboost as xgb

from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ================================
# Flags
# ================================

RUN_LGBM = True
RUN_XGB = False
RUN_BLEND = False

# ================================
# Load Data
# ================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Follow notebook: use last column as target
target = train.columns.tolist()[-1]

# Make working copies like notebook
df = train.copy()
df_test = test.copy()

# Feature engineering helpers (from notebook)
def create_frequency_features(df: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # Frequency encoding
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # Quantile binning for numeric columns
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(
                        df_test[col], bins=bins, labels=False, include_lowest=True
                    )
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)
    return df, df_test

from sklearn.model_selection import KFold

def target_encoding(df: pd.DataFrame, df_test: pd.DataFrame, n_splits: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean_features_train = pd.DataFrame(index=df.index)
    mean_features_test = pd.DataFrame(index=df_test.index)
    global_mean_all = df[target].mean()
    alpha = 50.0  # smoothing strength

    for col in cols:
        mean_encoded = np.zeros(len(df), dtype=float)
        for tr_idx, val_idx in kf.split(df):
            tr_fold = df.iloc[tr_idx]
            val_fold = df.iloc[val_idx]
            grp = tr_fold.groupby(col)[target]
            mean_map = grp.mean()
            cnt_map = grp.size()
            # smoothed mean = (mean*cnt + global*alpha) / (cnt + alpha)
            smoothed = (mean_map * cnt_map + global_mean_all * alpha) / (cnt_map + alpha)
            mean_encoded[val_idx] = val_fold[col].map(smoothed).fillna(global_mean_all)
        mean_features_train[f"mean_{col}"] = mean_encoded

        # Test mapping from full train (smoothed)
        grp_full = df.groupby(col)[target]
        mean_map_full = grp_full.mean()
        cnt_map_full = grp_full.size()
        smoothed_full = (mean_map_full * cnt_map_full + global_mean_all * alpha) / (cnt_map_full + alpha)
        mean_features_test[f"mean_{col}"] = df_test[col].map(smoothed_full).fillna(global_mean_all)

    df = pd.concat([df, mean_features_train], axis=1)
    df_test = pd.concat([df_test, mean_features_test], axis=1)
    df = df.copy()
    df_test = df_test.copy()
    return df, df_test

# Additional rounding and domain features like notebook
for c in ["annual_income", "loan_amount"]:
    for s, l in {"1s": 0, "10s": -1}.items():
        for g in [df, df_test]:
            g[f"{c}_ROUND_{s}"] = g[c].round(l).astype(int)
for g in [df, df_test]:
    g["subgrade"] = g["grade_subgrade"].str[1:].astype(int)
    g["grade"] = g["grade_subgrade"].str[0]
    g["total_debt_burden"] = (g["loan_amount"] * g["interest_rate"] / 100.0) / (g["annual_income"] + 1.0)

# Columns (recompute after new features)
cols = df.drop(columns=[target, "id"]).columns.tolist()
cat = [c for c in cols if df[c].dtype in ["object", "category"]]
num = [c for c in cols if df[c].dtype not in ["object", "category", "bool"]]

# Apply target encoding and frequency/bin features
df, df_test = target_encoding(df, df_test)
df, df_test = create_frequency_features(df, df_test)

# Prepare categorical dtype
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Drop columns list from notebook
remove = [
    'annual_income_ROUND_10s_bin10','annual_income_ROUND_1s_bin10','annual_income_ROUND_1s_bin15','annual_income_ROUND_1s_bin5',
    'annual_income_bin10','annual_income_bin5','credit_score_bin10','credit_score_bin5','debt_to_income_ratio_bin15','debt_to_income_ratio_bin5',
    'education_level_freq','gender_freq','interest_rate_bin10','interest_rate_bin5','loan_amount_ROUND_10s_bin5','loan_amount_ROUND_1s_bin10',
    'loan_amount_ROUND_1s_bin15','loan_amount_ROUND_1s_bin5','loan_amount_bin10','loan_amount_bin15','loan_amount_bin5','marital_status_freq',
    'subgrade','subgrade_bin10','subgrade_bin15','subgrade_bin5','subgrade_freq'
]
df, df_test = df.drop(columns=remove + ["id"], errors="ignore"), df_test.drop(columns=remove, errors="ignore")

# Final matrices
X = df.drop(columns=[target])
y = df[target]

# ================================
# LightGBM params from notebook
# ================================

def _build_lgbm_params(scale_pos_weight: float) -> dict[str, Any]:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 50,
        "learning_rate": 0.03,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "device": "cpu",
        "verbose": -1,
    }
    return params

# ================================
# LightGBM with CV Tuning (Notebook style)
# ================================

if RUN_LGBM:
    print("Training LightGBM with CV tuning")

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = _build_lgbm_params(scale_pos_weight)

    # Prepare data for CV (on full engineered matrix)
    lgb_train = lgb.Dataset(X, label=y, free_raw_data=True)

    # Run CV like notebook (7-fold, up to 20000 rounds with early stopping)
    cv_results = lgb.cv(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=20000,
        nfold=7,
        stratified=True,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
        seed=42,
    )

    # Resolve best iteration from keys like 'valid auc-mean'
    mean_keys = [k for k in cv_results.keys() if k.endswith("-mean")]
    if not mean_keys:
        mean_keys = list(cv_results.keys())
    best_round = len(cv_results[mean_keys[0]])
    print("Best iteration from CV:", best_round)

    # Set final estimators with buffer like notebook (+300)
    lgb_params["n_estimators"] = best_round + 300
    print("Using final n estimators:", lgb_params["n_estimators"])

    # Final train on full data with seed ensemble
    seeds = [42, 7, 99, 2025, 123]
    preds_list = []
    for s in seeds:
        params_s = dict(lgb_params)
        params_s["random_state"] = s
        model_s = lgb.LGBMClassifier(**params_s)
        model_s.fit(X, y)
        preds_list.append(model_s.predict_proba(df_test.drop(columns="id"))[:, 1])
    preds_mean = np.mean(np.vstack(preds_list), axis=0)
    sub = pd.DataFrame({"id": df_test["id"], target: preds_mean})
    sub.to_csv("submission_lgbm.csv", index=False)
    print("Saved submission_lgbm.csv (5-seed ensemble)")

# ================================
# XGBoost model
# ================================

if RUN_XGB:
    print("Training XGBoost")

    X_train_enc = pd.get_dummies(X_train)
    X_valid_enc = pd.get_dummies(X_valid)
    X_test_enc = pd.get_dummies(X_test)
    X_train_enc, X_valid_enc = X_train_enc.align(X_valid_enc, join="left", axis=1, fill_value=0)

    xgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.06,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "random_state": 42,
        "tree_method": "gpu_hist",
        "eval_metric": "auc",
    }

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train_enc, y_train)

    preds_valid_xgb = model_xgb.predict_proba(X_valid_enc)[:, 1]
    auc_xgb = roc_auc_score(y_valid, preds_valid_xgb)
    print("XGB validation AUC:", auc_xgb)

# ================================
# Blending
# ================================

if RUN_BLEND and RUN_LGBM and RUN_XGB:
    print("Running blend search")

    best_weight = 0
    best_auc = 0

    for w in np.linspace(0, 1, 21):
        blend = w * preds_valid_lgb + (1 - w) * preds_valid_xgb
        auc = roc_auc_score(y_valid, blend)

        if auc > best_auc:
            best_auc = auc
            best_weight = w

        print(f"Weight {w:.2f} AUC {auc:.5f}")

    print("Best blend weight:", best_weight)
    print("Best blend AUC:", best_auc)
