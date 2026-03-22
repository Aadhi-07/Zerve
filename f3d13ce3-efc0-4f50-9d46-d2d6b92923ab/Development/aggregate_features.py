def add_feats(df):
    d = df.copy()
    d["bin_sum"]  = d[bin_cols].sum(axis=1)
    d["num_mean"] = d[num_cols].mean(axis=1)
    d["num_std"]  = d[num_cols].std(axis=1)
    d["num_max"]  = d[num_cols].max(axis=1)
    d["num_min"]  = d[num_cols].min(axis=1)
    return d
 
Xr = add_feats(Xr)
Xt = add_feats(Xt)
print(Xr.shape)