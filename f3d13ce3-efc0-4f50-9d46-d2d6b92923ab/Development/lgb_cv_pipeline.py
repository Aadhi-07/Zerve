import subprocess, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Install LightGBM ──────────────────────────────────────────────────────
subprocess.run(
    ["pip", "install", "lightgbm", "--target=/tmp/lgb_pkg", "--quiet"],
    capture_output=True, text=True,
)
if "/tmp/lgb_pkg" not in sys.path:
    sys.path.insert(0, "/tmp/lgb_pkg")

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

print(f"LightGBM {lgb.__version__}")

# ── Load raw CSVs ─────────────────────────────────────────────────────────
lgb_tr = pd.read_csv("training_data.csv")
lgb_te = pd.read_csv("test_data_hackathon.csv")

print(f"Train: {lgb_tr.shape}   Test: {lgb_te.shape}")

# ── Feature columns (exclude id and target) ───────────────────────────────
feat_cols = [c for c in lgb_tr.columns if c not in ("id", "target")]
lgb_y = lgb_tr["target"].values

# ── Identify high-null columns (threshold: feature_39 ~69% missing) ───────
_null_pct = lgb_tr[feat_cols].isnull().mean()
_null_threshold = _null_pct["feature_39"]  # 0.69 – anchor on feature_39
high_null_cols = list(_null_pct[_null_pct >= _null_threshold].index)
print(f"High-null columns (>= {_null_threshold:.1%}): {high_null_cols}")

# ── Compute medians on training set, fill both train & test ───────────────
_medians = lgb_tr[feat_cols].median()
lgb_tr[feat_cols] = lgb_tr[feat_cols].fillna(_medians)
lgb_te[feat_cols] = lgb_te[feat_cols].fillna(_medians)

# ── Add missing indicators for high-null columns ──────────────────────────
# Rebuild indicators from fresh load (efficient: load id + high_null_cols only)
_tr_raw = pd.read_csv("training_data.csv", usecols=["id"] + high_null_cols)
_te_raw = pd.read_csv("test_data_hackathon.csv", usecols=["id"] + high_null_cols)

for _col in high_null_cols:
    _ind_col = f"{_col}_missing"
    lgb_tr[_ind_col] = _tr_raw[_col].isnull().astype(np.int8).values
    lgb_te[_ind_col] = _te_raw[_col].isnull().astype(np.int8).values

# Final feature set: original features + missing indicators
indicator_cols = [f"{c}_missing" for c in high_null_cols]
all_feat_cols = feat_cols + indicator_cols

lgb_Xr = lgb_tr[all_feat_cols].copy()
lgb_Xt = lgb_te[all_feat_cols].copy()

print(f"Feature matrix → train: {lgb_Xr.shape}  test: {lgb_Xt.shape}")
print(f"Missing indicator columns added: {indicator_cols}")

# ── Class imbalance (info only) ───────────────────────────────────────────
lgb_neg = int((lgb_y == 0).sum())
lgb_pos = int((lgb_y == 1).sum())
lgb_spw = lgb_neg / lgb_pos
print(f"Class ratio  neg={lgb_neg}  pos={lgb_pos}  base_rate={lgb_pos/(lgb_neg+lgb_pos):.3%}")
print(f"Using is_unbalance=True (LightGBM handles imbalance internally)")

# ── Model parameters ──────────────────────────────────────────────────────
lgb_params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# ── 5-fold Stratified CV ──────────────────────────────────────────────────
lgb_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof  = np.zeros(len(lgb_Xr))
lgb_tprd = np.zeros(len(lgb_Xt))

for _fold, (_tri, _vli) in enumerate(lgb_skf.split(lgb_Xr, lgb_y), 1):
    _dtrain = lgb.Dataset(lgb_Xr.iloc[_tri], label=lgb_y[_tri])
    _dval   = lgb.Dataset(lgb_Xr.iloc[_vli], label=lgb_y[_vli], reference=_dtrain)

    _booster = lgb.train(
        lgb_params, _dtrain,
        num_boost_round=1000,
        valid_sets=[_dval],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    lgb_oof[_vli]  = _booster.predict(lgb_Xr.iloc[_vli])
    lgb_tprd      += _booster.predict(lgb_Xt) / 5
    _fold_auc = average_precision_score(lgb_y[_vli], lgb_oof[_vli])
    print(f"fold {_fold}  pr-auc: {_fold_auc:.4f}  trees: {_booster.best_iteration}")

# ── OOF score ─────────────────────────────────────────────────────────────
lgb_oof_pr_auc = average_precision_score(lgb_y, lgb_oof)
print(f"\nmean OOF pr-auc : {lgb_oof_pr_auc:.4f}")
print(f"OOF  pred  mean : {lgb_oof.mean():.4f}   min: {lgb_oof.min():.4f}   max: {lgb_oof.max():.4f}")
print(f"test pred  mean : {lgb_tprd.mean():.4f}   min: {lgb_tprd.min():.4f}   max: {lgb_tprd.max():.4f}")
print(f"True base rate  : {lgb_y.mean():.4f}")

# Sanity check: all predictions in [0, 1]
assert lgb_tprd.min() >= 0.0 and lgb_tprd.max() <= 1.0, "Predictions out of [0,1] range!"

# ── Submission ────────────────────────────────────────────────────────────
lgb_sub = pd.DataFrame({"id": lgb_te["id"], "target": lgb_tprd})
lgb_sub.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv saved  shape={lgb_sub.shape}  columns={list(lgb_sub.columns)}")
print(lgb_sub.head())
