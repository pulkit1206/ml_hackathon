"""Train + compare the 5 models (or only the ones named).

    python scripts/train_models.py                      # all five
    python scripts/train_models.py RandomForest LightGBM
    python scripts/train_models.py --all-features LightGBM   # ignore the feature selection
"""
import sys

import pandas as pd

from common import TEST_PQ, TRAIN_PQ, selected_features
from harness import MODEL_ZOO, run_experiment
from pathsnap.features import FEATURES_ALL

args = sys.argv[1:]
feats = FEATURES_ALL if "--all-features" in args else selected_features()
names = [a for a in args if not a.startswith("--")] or list(MODEL_ZOO)
train, test = pd.read_parquet(TRAIN_PQ), pd.read_parquet(TEST_PQ)
print(f"train: {len(train):,} rows / {train.trip_id.nunique()} trips | test: {len(test):,} rows / "
      f"{test.trip_id.nunique()} trips | features ({len(feats)}): {feats}\n")
for n in names:
    fam, params, build, cap = MODEL_ZOO[n]
    tag = n + ("_allfeat" if "--all-features" in args else "")
    run_experiment(tag, fam, lambda: build(params), params, train, test, feats, cap)
