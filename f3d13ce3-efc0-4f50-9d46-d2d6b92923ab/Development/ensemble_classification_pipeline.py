import pandas as pd, numpy as np, warnings, copy
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV

tr = pd.read_csv("training_data.csv")
te = pd.read_csv("test_data_hackathon.csv")
print(tr["target"].value_counts(normalize=True).round(4))

bin_cols = ["feature_4","feature_5","feature_6","feature_11","feature_14","feature_16","feature_18","feature_19","feature_20","feature_21","feature_22","feature_27","feature_30","feature_32","feature_41","feature_44","feature_46"]
cat_cols = ["feature_3","feature_7","feature_8","feature_12","feature_15","feature_23","feature_25","feature_28","feature_31","feature_34","feature_35","feature_39","feature_42","feature_49"]
num_cols = ["feature_1","feature_2","feature_9","feature_10","feature_13","feature_17","feature_24","feature_26","feature_29","feature_33","feature_36","feature_37","feature_38","feature_40","feature_43","feature_45","feature_47","feature_48","feature_50"]
all_cols = num_cols + bin_cols + cat_cols

Xr = tr[all_cols].copy()
Xt = te[all_cols].copy()
y  = tr["target"].values

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
Xr[cat_cols] = enc.fit_transform(Xr[cat_cols].astype(str))
Xt[cat_cols] = enc.transform(Xt[cat_cols].astype(str))

for c in num_cols + bin_cols:
    m = Xr[c].median()
    Xr[c] = Xr[c].fillna(m)
    Xt[c] = Xt[c].fillna(m)

Xr["bin_sum"]  = Xr[bin_cols].sum(axis=1);  Xt["bin_sum"]  = Xt[bin_cols].sum(axis=1)
Xr["num_mean"] = Xr[num_cols].mean(axis=1); Xt["num_mean"] = Xt[num_cols].mean(axis=1)
Xr["num_std"]  = Xr[num_cols].std(axis=1);  Xt["num_std"]  = Xt[num_cols].std(axis=1)
Xr["num_max"]  = Xr[num_cols].max(axis=1);  Xt["num_max"]  = Xt[num_cols].max(axis=1)

neg = (y==0).sum()
pos = (y==1).sum()
spw = neg/pos
print(f"scale_pos_weight: {spw:.1f}")

mdls = {
    "hgb": (HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=1.0,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30
    ), 0.6),
    "rf": (RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ), 0.4)
}

skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof  = {n: np.zeros(len(Xr)) for n in mdls}
tprd = {n: np.zeros(len(Xt)) for n in mdls}

for name, (base, _) in mdls.items():
    print(f"\n[{name}]")
    for fold, (tri, vli) in enumerate(skf.split(Xr, y), 1):
        m = copy.deepcopy(base)
        m.fit(Xr.iloc[tri], y[tri])
        oof[name][vli] = m.predict_proba(Xr.iloc[vli])[:, 1]
        tprd[name]    += m.predict_proba(Xt)[:, 1] / 5
        print(f"  fold {fold}  pr-auc: {average_precision_score(y[vli], oof[name][vli]):.4f}")
    print(f"  oof: {average_precision_score(y, oof[name]):.4f}")

final_oof  = sum(oof[n] * w for n, (_, w) in mdls.items())
final_pred = sum(tprd[n] * w for n, (_, w) in mdls.items())
print(f"\nensemble oof pr-auc: {average_precision_score(y, final_oof):.4f}")
print(f"pred mean: {final_pred.mean():.4f}  min: {final_pred.min():.4f}  max: {final_pred.max():.4f}")

sub = pd.DataFrame({"id": te["id"], "target": final_pred})
sub.to_csv("submission2.csv", index=False)
print(sub.head(10))