"""Experiment harness: group-aware CV on the 80 training trips + evaluation on the 10 student-test trips.
Every run appends one row to results/models/experiments.csv (and per-fold rows to experiments_folds.csv)."""
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS as OUT, N_SPLITS, SEED
from pathsnap.metrics import candidate_metrics


def subsample_points(df, n_rows):
    """Subsample whole GPS points (with all their candidates), never single rows."""
    if not n_rows or len(df) <= n_rows:
        return df
    keys = df[["trip_id", "point_idx"]].drop_duplicates()
    keep = keys.sample(frac=n_rows / len(df), random_state=SEED)
    return df.merge(keep, on=["trip_id", "point_idx"])


# name: (family, initial hyper-parameters, builder, training-row cap for slow models)
MODEL_ZOO = {
    "LogisticRegression": ("Linear", {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p, random_state=SEED)), None),
    "KNN": ("Instance-based", {"n_neighbors": 25, "weights": "distance"},
        lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p, n_jobs=-1)), 200_000),
    "RandomForest": ("Ensemble (bagging)", {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                                            "class_weight": "balanced"},
        lambda p: RandomForestClassifier(**p, n_jobs=-1, random_state=SEED), None),
    "LightGBM": ("Boosting", {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "is_unbalance": True},
        lambda p: LGBMClassifier(**p, random_state=SEED, verbose=-1), None),
    "MLP": ("Neural network", {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "early_stopping": True, "max_iter": 50},
        lambda p: make_pipeline(StandardScaler(), MLPClassifier(
            **{**p, "hidden_layer_sizes": tuple(p["hidden_layer_sizes"])}, random_state=SEED)), 500_000),
}


def _append(path, row_df):
    if path.exists():
        old = pd.read_csv(path)
        row_df = pd.concat([old, row_df], ignore_index=True)
    row_df.to_csv(path, index=False)


def run_experiment(name, family, make_model, params, train, test, features, train_subsample=None):
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id)):
        tr_df, va_df = subsample_points(train.iloc[tr], train_subsample), train.iloc[va]
        m = make_model()
        t0 = time.perf_counter(); m.fit(tr_df[features], tr_df.label); fit_s = time.perf_counter() - t0
        mt = candidate_metrics(tr_df, m.predict_proba(tr_df[features])[:, 1])
        mv = candidate_metrics(va_df, m.predict_proba(va_df[features])[:, 1])
        folds.append({"model": name, "fold": k, "fit_s": fit_s, **{f"train_{a}": b for a, b in mt.items()},
                      **{f"val_{a}": b for a, b in mv.items()}})
        print(f"  [{name}] fold {k}: val top1={mv['top1']:.4f} macroF1={mv['macro_f1']:.4f} (train top1={mt['top1']:.4f})")
    f = pd.DataFrame(folds)

    fit_df = subsample_points(train, train_subsample)
    m = make_model()
    t0 = time.perf_counter(); m.fit(fit_df[features], fit_df.label); train_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(test[features])[:, 1]; inf = time.perf_counter() - t0
    te = candidate_metrics(test, p)
    (OUT / "preds").mkdir(exist_ok=True)
    test[["trip_id", "point_idx", "arc_id", "label"]].assign(proba=p).to_parquet(OUT / "preds" / f"{name}.parquet", index=False)

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": name, "family": family,
           "params": json.dumps(params), "seed": SEED, "n_splits": N_SPLITS, "n_features": len(features),
           "features": ",".join(features), "train_rows_used": len(fit_df),
           "cv_top1_mean": f.val_top1.mean(), "cv_top1_std": f.val_top1.std(),
           "cv_macro_f1_mean": f.val_macro_f1.mean(), "cv_macro_f1_std": f.val_macro_f1.std(),
           "cv_log_loss_mean": f.val_log_loss.mean(), "cv_train_top1_mean": f.train_top1.mean(),
           **{f"test_{k}": v for k, v in te.items()},
           "train_time_s": train_time, "inference_ms_per_1k_rows": 1e6 * inf / len(test)}
    _append(OUT / "experiments.csv", pd.DataFrame([row]))
    _append(OUT / "experiments_folds.csv", f)
    print(f"==> {name}: CV top1={row['cv_top1_mean']:.4f}±{row['cv_top1_std']:.4f} | test top1={te['top1']:.4f} "
          f"macroF1={te['macro_f1']:.4f} logloss={te['log_loss']:.4f}\n")
    return m, row
