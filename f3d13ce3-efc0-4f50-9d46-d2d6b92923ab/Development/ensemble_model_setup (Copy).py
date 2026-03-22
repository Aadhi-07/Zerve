

# hgb = HistGradientBoostingClassifier(
#     max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
#     min_samples_leaf=30, l2_regularization=1.0,
#     class_weight="balanced", random_state=42,
#     early_stopping=True, validation_fraction=0.1,
#     n_iter_no_change=30, scoring="average_precision"
# )

# rf = RandomForestClassifier(
#     n_estimators=300, max_depth=12, min_samples_leaf=20,
#     class_weight="balanced_subsample", n_jobs=-1, random_state=42
# )

# lgbm = lgb.LGBMClassifier(
#     objective="binary",
#     metric="average_precision",
#     learning_rate=0.03,
#     n_estimators=3000,
#     num_leaves=96,
#     subsample=0.85,
#     colsample_bytree=0.7,
#     min_child_samples=120,
#     reg_alpha=1.0,
#     reg_lambda=3.0,
#     scale_pos_weight=6,
#     random_state=42,
#     n_jobs=-1
# )

# mdls = {"hgb": (hgb, 0.6), "rf": (rf, 0.4)}
# print(mdls)