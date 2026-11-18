"""Permutation importance (drop in validation Top-1 when one feature is shuffled), 3-fold GroupKFold on
the TRAINING trips only, using LightGBM with its initial settings. Also LightGBM gain importance.

Selection rule (fixed before looking at the result): keep a feature if shuffling it lowers
validation Top-1 by at least 0.001 on average. Writes results/models/selected_features.json.

    python scripts/feature_importance.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common import MODELS, N_SPLITS, PLOTS, SEED, TRAIN_PQ
from pathsnap.features import FEATURES_ALL
from pathsnap.metrics import top1_accuracy

MIN_DROP = 0.001
train = pd.read_parquet(TRAIN_PQ)
rng = np.random.default_rng(SEED)
drops, gains = {f: [] for f in FEATURES_ALL}, np.zeros(len(FEATURES_ALL))
for k, (tr, va) in enumerate(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id)):
    tr_df, va_df = train.iloc[tr], train.iloc[va].copy()
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, is_unbalance=True,
                       random_state=SEED, verbose=-1).fit(tr_df[FEATURES_ALL], tr_df.label)
    gains += m.booster_.feature_importance("gain")
    base = top1_accuracy(va_df, m.predict_proba(va_df[FEATURES_ALL])[:, 1])
    for f in FEATURES_ALL:
        keep = va_df[f].to_numpy().copy()
        va_df[f] = rng.permutation(keep)
        drops[f].append(base - top1_accuracy(va_df, m.predict_proba(va_df[FEATURES_ALL])[:, 1]))
        va_df[f] = keep
    print(f"fold {k}: base top1 = {base:.4f}")

res = pd.DataFrame({"feature": FEATURES_ALL,
                    "perm_top1_drop_mean": [np.mean(drops[f]) for f in FEATURES_ALL],
                    "perm_top1_drop_std": [np.std(drops[f]) for f in FEATURES_ALL],
                    "lgbm_gain_share": gains / gains.sum()}).sort_values("perm_top1_drop_mean", ascending=False)
res["selected"] = res.perm_top1_drop_mean >= MIN_DROP
res.to_csv(MODELS / "feature_importance.csv", index=False)
json.dump(res[res.selected].feature.tolist(), open(MODELS / "selected_features.json", "w"), indent=2)
print(res.to_string(index=False))
print("\nselected:", res[res.selected].feature.tolist())

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(res.feature, res.perm_top1_drop_mean, xerr=res.perm_top1_drop_std,
        color=["#3b6ea5" if s else "#bbbbbb" for s in res.selected])
ax.axvline(MIN_DROP, color="k", ls="--", lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("Drop in validation Top-1 when the feature is shuffled")
ax.set_title("Permutation importance (LightGBM, GroupKFold on train; grey = dropped)")
plt.tight_layout(); plt.savefig(PLOTS / "feature_importance.png", dpi=150)
