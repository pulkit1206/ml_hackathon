"""Real per-round training/validation log-loss curves for LightGBM (one GroupKFold split of the training trips).
Random Forest / KNN / Logistic Regression have no epoch-wise history, so no curve is drawn for them.

    python scripts/lgbm_curves.py                 # initial settings
    python scripts/lgbm_curves.py tuned           # Optuna best settings (artifacts/best_params.json)
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common import ART, MODELS, N_SPLITS, PLOTS, SEED, TRAIN_PQ, selected_features
from harness import MODEL_ZOO

warnings.filterwarnings("ignore")
tag = sys.argv[1] if len(sys.argv) > 1 else "initial"
params = json.load(open(ART / "best_params.json")) if tag == "tuned" else MODEL_ZOO["LightGBM"][1]
feats = selected_features()
train = pd.read_parquet(TRAIN_PQ)
tr, va = next(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id))
a, b = train.iloc[tr], train.iloc[va]
m = LGBMClassifier(**params, random_state=SEED, verbose=-1)
m.fit(a[feats], a.label, eval_set=[(a[feats], a.label), (b[feats], b.label)],
      eval_names=["train", "validation"], eval_metric="binary_logloss")
h = pd.DataFrame({k: v["binary_logloss"] for k, v in m.evals_result_.items()})
h.to_csv(MODELS / f"lgbm_loss_curve_{tag}.csv", index_label="round")
plt.figure(figsize=(7, 4))
plt.plot(h["train"], label="training trips"); plt.plot(h["validation"], label="validation trips (unseen)")
plt.xlabel("boosting round"); plt.ylabel("binary log loss"); plt.title(f"LightGBM ({tag}) loss per boosting round")
plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / f"lgbm_loss_curve_{tag}.png", dpi=150)
print(f"best validation round: {h['validation'].idxmin()} (loss {h['validation'].min():.4f}); "
      f"final train {h['train'].iloc[-1]:.4f} vs validation {h['validation'].iloc[-1]:.4f}")
