oof_blend = 0
tst_blend = 0

for name, (_, w) in mdls.items():
    oof_blend += w * oof[name]
    tst_blend += w * tprd[name]

print("\nFINAL BLEND PR-AUC:",
      average_precision_score(y, oof_blend))