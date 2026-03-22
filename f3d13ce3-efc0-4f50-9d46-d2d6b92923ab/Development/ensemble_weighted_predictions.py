# import pandas as pd
# import numpy as np
# from sklearn.metrics import average_precision_score

final_oof  = sum(oof[n]  * w for n, (_, w) in mdls.items())
final_pred = sum(tprd[n] * w for n, (_, w) in mdls.items())

print(f"ensemble oof pr-auc: {average_precision_score(y, final_oof):.4f}")

sub = pd.DataFrame({"id": te["id"], "target": final_pred})
print(sub)
sub.to_csv("submission.csv", index=False)
print(sub.head())