import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

# Use pre-computed weighted blend from python_block_10 upstream
# oof_blend and tst_blend are the weighted ensemble OOF/test predictions
final_oof  = oof_blend
final_pred = tst_blend

# Evaluate ensemble on out-of-fold predictions
oof_pr_auc = average_precision_score(y, final_oof)
print(f"Ensemble OOF PR-AUC: {oof_pr_auc:.4f}")

# Build submission DataFrame
sub = pd.DataFrame({"id": te["id"], "target": final_pred})
sub.to_csv("submission.csv", index=False)
print(f"\nSubmission shape: {sub.shape}")
print(sub.head(10))
print(f"\nPrediction stats — min: {final_pred.min():.4f}, max: {final_pred.max():.4f}, mean: {final_pred.mean():.4f}")
