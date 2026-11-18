"""Evaluate the tuned LightGBM with the same harness, and export the final model.

    python scripts/train_tuned.py
"""
import json
from datetime import datetime

import joblib
import lightgbm
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier

from common import ART, MODELS, SEED, TEST_PQ, TRAIN_PQ, selected_features
from harness import run_experiment

best = json.load(open(ART / "best_params.json"))
feats = selected_features()
train, test = pd.read_parquet(TRAIN_PQ), pd.read_parquet(TEST_PQ)
make = lambda: LGBMClassifier(**best, random_state=SEED, verbose=-1)
model, _ = run_experiment("LightGBM_tuned", "Boosting", make, best, train, test, feats)

exp = pd.read_csv(MODELS / "experiments.csv")
cmp_ = exp[(exp.features == ",".join(feats)) & exp.model.isin(["LightGBM", "LightGBM_tuned"])]
cmp_ = cmp_.sort_values("timestamp").groupby("model").tail(1)
cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std", "test_top1", "test_macro_f1",
        "test_log_loss", "train_time_s", "inference_ms_per_1k_rows"]
cmp_[cols].to_csv(MODELS / "tuned_vs_untuned.csv", index=False)
print(cmp_[cols].to_string(index=False))

joblib.dump(model, ART / "final_model.joblib")
json.dump(feats, open(ART / "feature_columns.json", "w"), indent=2)
json.dump({"model": "LightGBM (Optuna-tuned)", "trained_on": "80 training trips (labelled points)",
           "date": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__,
           "usage": "Matcher('artifacts').predict_trip(folder)  or  model.predict_proba(df[features])[:, 1]"},
          open(ART / "model_info.json", "w"), indent=2)
print("exported artifacts/final_model.joblib + feature_columns.json + model_info.json")
