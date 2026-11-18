"""
Feature selection on the TRAINING trips only.
Permutation importance measured as the drop in Top-1 accuracy on held-out GroupKFold folds,
plus LightGBM gain importance.

Usage:
    python feature_importance.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from train_models import FEATURES, SEED, TRAIN, top1_accuracy

train = pd.read_parquet(TRAIN)
rng = np.random.default_rng(SEED)
drops = {f: [] for f in FEATURES}
gains = np.zeros(len(FEATURES))

for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(train, groups=train.trip_id)):
    tr_df, va_df = train.iloc[tr], train.iloc[va].copy()
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                       is_unbalance=True, random_state=SEED, verbose=-1)
    m.fit(tr_df[FEATURES], tr_df.label)
    gains += m.booster_.feature_importance("gain")
    base = top1_accuracy(va_df, m.predict_proba(va_df[FEATURES])[:, 1])
    for f in FEATURES:
        saved = va_df[f].to_numpy().copy()
        va_df[f] = rng.permutation(saved)
        drops[f].append(base - top1_accuracy(va_df, m.predict_proba(va_df[FEATURES])[:, 1]))
        va_df[f] = saved
    print(f"fold {k}: base top1={base:.4f}")

res = pd.DataFrame({"feature": FEATURES,
                    "perm_top1_drop_mean": [np.mean(drops[f]) for f in FEATURES],
                    "perm_top1_drop_std": [np.std(drops[f]) for f in FEATURES],
                    "lgbm_gain": gains / gains.sum()}).sort_values("perm_top1_drop_mean", ascending=False)
res.to_csv("results/feature_importance.csv", index=False)
print(res.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(res.feature, res.perm_top1_drop_mean, xerr=res.perm_top1_drop_std, color="#3b6ea5")
ax.invert_yaxis()
ax.set_xlabel("Drop in validation Top-1 when feature is shuffled")
ax.set_title("Permutation importance (LightGBM, 3-fold GroupKFold on train)")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
