


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

print(f"Xr shape: {Xr.shape}, Xt shape: {Xt.shape}")
print(f"y distribution — 0: {(y == 0).sum()}, 1: {(y == 1).sum()}")
print(f"Missing values in Xr: {Xr.isnull().sum().sum()}, Xt: {Xt.isnull().sum().sum()}")
