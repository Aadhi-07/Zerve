
hgb = HistGradientBoostingClassifier(
    max_iter = 420,
    learning_rate = 0.04,
    max_leaf_nodes = 63,
    min_samples_leaf = 40,
    l2_regularization = 2.0,
    class_weight = "balanced",
    early_stopping = False,     # IMPORTANT
    random_state = 42
)

rf = RandomForestClassifier(
    n_estimators = 220,
    max_depth = 12,
    min_samples_leaf = 25,
    class_weight = "balanced_subsample",
    n_jobs = -1,
    random_state = 42
)
mdls = {
    "hgb": (hgb, 0.65),
    "rf":  (rf, 0.35)
}

print(mdls)