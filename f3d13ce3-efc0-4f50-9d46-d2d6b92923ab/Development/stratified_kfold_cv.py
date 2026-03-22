
print(mdls)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof = {n: np.zeros(len(Xr)) for n in mdls}
tprd = {n: np.zeros(len(Xt)) for n in mdls}

for name, (base, _) in mdls.items():
    print(f"\n{name}")

    for fold, (tri, vli) in enumerate(skf.split(Xr, y), 1):
        m = copy.deepcopy(base)
        m.fit(Xr.iloc[tri], y[tri])

        oof[name][vli] = m.predict_proba(Xr.iloc[vli])[:, 1]
        tprd[name] += m.predict_proba(Xt)[:, 1] / skf.n_splits

        sc = average_precision_score(y[vli], oof[name][vli])
        print(f"fold{fold}:{sc:.4f}", end=" ")
    print("\nOOF:", average_precision_score(y, oof[name]))
