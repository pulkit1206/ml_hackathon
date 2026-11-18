"""Bayesian hyper-parameter optimisation of LightGBM with Optuna (TPE sampler).

Objective: mean validation Top-1 over 3-fold GroupKFold on the 80 TRAINING trips (student_test is not used).
All trials are stored in results/models/optuna.db (resumable) and exported to optuna_trials.csv.

    python scripts/optuna_search.py          # 40 trials
    python scripts/optuna_search.py 60
"""
import json
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from common import ART, MODELS, N_SPLITS, PLOTS, SEED, TRAIN_PQ, selected_features
from pathsnap.metrics import top1_accuracy

warnings.filterwarnings("ignore")
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
feats = selected_features()
train = pd.read_parquet(TRAIN_PQ)
folds = list(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id))


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),          # tree complexity
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),  # step size
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),                      # row sampling
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),        # feature sampling
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),        # L2 regularisation
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),     # imbalance handling
    }
    tops, losses = [], []
    for k, (tr, va) in enumerate(folds):
        a, b = train.iloc[tr], train.iloc[va]
        m = LGBMClassifier(**params, random_state=SEED, verbose=-1).fit(a[feats], a.label)
        p = m.predict_proba(b[feats])[:, 1]
        tops.append(top1_accuracy(b, p)); losses.append(log_loss(b.label, np.clip(p, 1e-7, 1 - 1e-7)))
        trial.report(float(np.mean(tops)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("cv_top1_std", float(np.std(tops)))
    trial.set_user_attr("cv_log_loss", float(np.mean(losses)))
    return float(np.mean(tops))


study = optuna.create_study(study_name="lgbm_top1", direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
                            storage=f"sqlite:///{MODELS / 'optuna.db'}", load_if_exists=True)
# reference trial close to the untuned configuration
study.enqueue_trial({"num_leaves": 31, "max_depth": 16, "min_child_samples": 20, "learning_rate": 0.05,
                     "n_estimators": 400, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3,
                     "is_unbalance": True}, skip_if_exists=True)
t0 = time.perf_counter()
study.optimize(objective, n_trials=N_TRIALS)
secs = time.perf_counter() - t0

study.trials_dataframe().to_csv(MODELS / "optuna_trials.csv", index=False)
best = {**study.best_params, "subsample_freq": 1}
json.dump(best, open(ART / "best_params.json", "w"), indent=2)
done = [t for t in study.trials if t.state.name == "COMPLETE"]
json.dump({"trials_total": len(study.trials), "trials_completed": len(done),
           "trials_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
           "best_trial": study.best_trial.number, "best_cv_top1": study.best_value,
           "best_cv_log_loss": study.best_trial.user_attrs.get("cv_log_loss"),
           "search_seconds_this_run": round(secs, 1)}, open(MODELS / "optuna_summary.json", "w"), indent=2)
plot_optimization_history(study); plt.tight_layout(); plt.savefig(PLOTS / "optuna_history.png", dpi=150); plt.close()
plot_param_importances(study); plt.tight_layout(); plt.savefig(PLOTS / "optuna_param_importance.png", dpi=150); plt.close()
print(f"\nbest CV top1 {study.best_value:.4f} (trial {study.best_trial.number}); {secs/60:.1f} min")
print(json.dumps(best, indent=2))
