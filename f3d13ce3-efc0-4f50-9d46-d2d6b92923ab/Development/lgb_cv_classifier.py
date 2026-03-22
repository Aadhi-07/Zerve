import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")

# Add the /tmp install path so lightgbm is importable in this block
if "/tmp/lgb_pkg" not in sys.path:
    sys.path.insert(0, "/tmp/lgb_pkg")

import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

print(f"LightGBM version: {lgb.__version__}")

# ── Load data ──────────────────────────────────────────────────────────────
lgb_tr = pd.read_csv("training_data.csv")
lgb_te = pd.read_csv("test_data_hackathon.csv")

lgb_bin_cols = ["feature_4","feature_5","feature_6","feature_11","feature_14","feature_16","feature_18","feature_19","feature_20","feature_21","feature_22","feature_27","feature_30","feature_32","feature_41","feature_44","feature_46"]
lgb_cat_cols = ["feature_3","feature_7","feature_8","feature_12","feature_15","feature_23","feature_25","feature_28","feature_31","feature_34","feature_35","feature_39","feature_42","feature_49"]
lgb_num_cols = ["feature_1","feature_2","feature_9","feature_10","feature_13","feature_17","feature_24","feature_26","feature_29","feature_33","feature_36","feature_37","feature_38","feature_40","feature_43","feature_45","feature_47","feature_48","feature_50"]
lgb_all_cols = lgb_num_cols + lgb_bin_cols + lgb_cat_cols

lgb_Xr = lgb_tr[lgb_all_cols].copy()
lgb_Xt = lgb_te[lgb_all_cols].copy()
lgb_y  = lgb_tr["target"].values

# ── Encode & impute ────────────────────────────────────────────────────────
lgb_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
lgb_Xr[lgb_cat_cols] = lgb_enc.fit_transform(lgb_Xr[lgb_cat_cols].astype(str))
lgb_Xt[lgb_cat_cols] = lgb_enc.transform(lgb_Xt[lgb_cat_cols].astype(str))

for _col in lgb_num_cols + lgb_bin_cols:
    _med = lgb_Xr[_col].median()
    lgb_Xr[_col] = lgb_Xr[_col].fillna(_med)
    lgb_Xt[_col] = lgb_Xt[_col].fillna(_med)

# ── Feature engineering ────────────────────────────────────────────────────
lgb_Xr["bin_sum"]  = lgb_Xr[lgb_bin_cols].sum(axis=1);  lgb_Xt["bin_sum"]  = lgb_Xt[lgb_bin_cols].sum(axis=1)
lgb_Xr["num_mean"] = lgb_Xr[lgb_num_cols].mean(axis=1); lgb_Xt["num_mean"] = lgb_Xt[lgb_num_cols].mean(axis=1)
lgb_Xr["num_std"]  = lgb_Xr[lgb_num_cols].std(axis=1);  lgb_Xt["num_std"]  = lgb_Xt[lgb_num_cols].std(axis=1)
lgb_Xr["num_max"]  = lgb_Xr[lgb_num_cols].max(axis=1);  lgb_Xt["num_max"]  = lgb_Xt[lgb_num_cols].max(axis=1)

# ── Class imbalance ────────────────────────────────────────────────────────
lgb_neg = (lgb_y == 0).sum()
lgb_pos = (lgb_y == 1).sum()
lgb_spw = lgb_neg / lgb_pos
print(f"scale_pos_weight: {lgb_spw:.1f}  (neg={lgb_neg}, pos={lgb_pos})")

# ── Model params ───────────────────────────────────────────────────────────
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
    "scale_pos_weight": lgb_spw,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1
}

# ── 5-fold CV ──────────────────────────────────────────────────────────────
lgb_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof  = np.zeros(len(lgb_Xr))
lgb_tprd = np.zeros(len(lgb_Xt))

for lgb_fold, (lgb_tri, lgb_vli) in enumerate(lgb_skf.split(lgb_Xr, lgb_y), 1):
    _dtrain = lgb.Dataset(lgb_Xr.iloc[lgb_tri], label=lgb_y[lgb_tri])
    _dval   = lgb.Dataset(lgb_Xr.iloc[lgb_vli], label=lgb_y[lgb_vli], reference=_dtrain)

    _booster = lgb.train(
        lgb_params, _dtrain,
        num_boost_round=1000,
        valid_sets=[_dval],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )

    lgb_oof[lgb_vli]  = _booster.predict(lgb_Xr.iloc[lgb_vli])
    lgb_tprd         += _booster.predict(lgb_Xt) / 5
    _fold_auc = average_precision_score(lgb_y[lgb_vli], lgb_oof[lgb_vli])
    print(f"fold {lgb_fold}  pr-auc: {_fold_auc:.4f}  trees: {_booster.best_iteration}")

# ── Final scores ───────────────────────────────────────────────────────────
lgb_oof_pr_auc = average_precision_score(lgb_y, lgb_oof)
print(f"\noof pr-auc: {lgb_oof_pr_auc:.4f}")
print(f"pred mean:  {lgb_tprd.mean():.4f}  min: {lgb_tprd.min():.4f}  max: {lgb_tprd.max():.4f}")

# ── Submission ─────────────────────────────────────────────────────────────
lgb_sub = pd.DataFrame({"id": lgb_te["id"], "target": lgb_tprd})
lgb_sub.to_csv("submission3.csv", index=False)
print(lgb_sub.head())
