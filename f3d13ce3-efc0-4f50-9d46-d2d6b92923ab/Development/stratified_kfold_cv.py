import copy
print("Model configs:", {n: w for n, (_, w) in mdls.items()})

# Use 2-fold CV with lighter models to avoid timeout on large dataset (476k rows)
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

oof  = {n: np.zeros(len(Xr)) for n in mdls}
tprd = {n: np.zeros(len(Xt)) for n in mdls}

for name, (base, _) in mdls.items():
    print(f"\nTraining: {name}")
    # Use lighter model overrides to avoid timeout
    _base = copy.deepcopy(base)
    if hasattr(_base, 'max_iter'):
        _base.max_iter = 100          # HGB: 100 iterations instead of 420
    if hasattr(_base, 'n_estimators'):
        _base.n_estimators = 60       # RF: 60 trees instead of 220

    for fold, (tri, vli) in enumerate(skf.split(Xr, y), 1):
        _m = copy.deepcopy(_base)
        _m.fit(Xr.iloc[tri], y[tri])

        oof[name][vli]  = _m.predict_proba(Xr.iloc[vli])[:, 1]
        tprd[name]     += _m.predict_proba(Xt)[:, 1] / skf.n_splits

        _sc = average_precision_score(y[vli], oof[name][vli])
        print(f"  fold{fold}: PR-AUC={_sc:.4f}", flush=True)

    print(f"  OOF PR-AUC: {average_precision_score(y, oof[name]):.4f}")

print("\nCV complete — oof and tprd ready for blending.")
