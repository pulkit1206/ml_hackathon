"""
Train and compare 5 models with group-aware CV (GroupKFold by trip).
Every run is appended to results/experiments.csv.

Usage:
    python train_models.py
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 3            # drop to 3 if you are short on time; write down that you did
TRAIN = "results/train_candidates.parquet"
TEST = "results/student_test_candidates.parquet"
FEATURES = ["dist", "dist_rank", "dist_minus_min", "n_cands", "heading_diff",
            "speed", "time_gap", "arc_len", "proj_frac", "conn_prev_nearest"]


# ---------- metrics ----------
def top1_accuracy(df, proba):
    """For each GPS point pick the highest-probability candidate; correct if its label is 1."""
    tmp = df[["trip_id", "point_idx", "label"]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby(["trip_id", "point_idx"]).p.idxmax()]
    return best.label.mean()


def all_metrics(df, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(df.label, pred, average="macro"),
        "log_loss": log_loss(df.label, np.clip(proba, 1e-7, 1 - 1e-7)),
        "top1": top1_accuracy(df, proba),
    }


# ---------- harness ----------
def run_experiment(name, family, make_model, params, train, test, features=FEATURES,
                   train_subsample=None, log_path="results/experiments.csv"):
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    for k, (tr, va) in enumerate(gkf.split(train, groups=train.trip_id)):
        tr_df, va_df = train.iloc[tr], train.iloc[va]
        if train_subsample and len(tr_df) > train_subsample:
            tr_df = tr_df.sample(train_subsample, random_state=SEED)
        m = make_model()
        m.fit(tr_df[features], tr_df.label)
        m_tr = all_metrics(tr_df, m.predict_proba(tr_df[features])[:, 1])
        m_va = all_metrics(va_df, m.predict_proba(va_df[features])[:, 1])
        fold_rows.append({"fold": k, **{f"train_{a}": b for a, b in m_tr.items()},
                          **{f"val_{a}": b for a, b in m_va.items()}})
        print(f"  [{name}] fold {k}: val top1={m_va['top1']:.4f} macroF1={m_va['macro_f1']:.4f}")
    folds = pd.DataFrame(fold_rows)

    # refit on all training trips, evaluate on student_test
    fit_df = train.sample(train_subsample, random_state=SEED) if train_subsample and len(train) > train_subsample else train
    m = make_model()
    t0 = time.perf_counter(); m.fit(fit_df[features], fit_df.label); train_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = m.predict_proba(test[features])[:, 1]; inf_time = time.perf_counter() - t0
    te = all_metrics(test, p)

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "model": name, "family": family,
           "params": json.dumps(params), "seed": SEED, "n_splits": N_SPLITS, "features": ",".join(features),
           "cv_macro_f1_mean": folds.val_macro_f1.mean(), "cv_macro_f1_std": folds.val_macro_f1.std(),
           "cv_top1_mean": folds.val_top1.mean(), "cv_top1_std": folds.val_top1.std(),
           "cv_log_loss_mean": folds.val_log_loss.mean(),
           "cv_train_top1_mean": folds.train_top1.mean(),   # compare with cv_top1_mean for over/underfitting
           "test_macro_f1": te["macro_f1"], "test_top1": te["top1"], "test_log_loss": te["log_loss"],
           "train_time_s": train_time, "inference_ms_per_1k": 1000 * inf_time / len(test) * 1000}
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    folds.assign(model=name).to_csv("results/experiments_folds.csv", mode="a",
                                    header=not os.path.exists("results/experiments_folds.csv"), index=False)
    print(f"==> {name}: CV top1={row['cv_top1_mean']:.4f}±{row['cv_top1_std']:.4f}  test top1={te['top1']:.4f}\n")
    return m, row


# ---------- the five models ----------
MODELS = {
    "LogisticRegression": ("Linear", {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p, random_state=SEED)), None),
    "KNN": ("Instance-based", {"n_neighbors": 25, "weights": "distance"},
        lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p, n_jobs=-1)), 200_000),
    "RandomForest": ("Ensemble (bagging)", {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5,
                                            "class_weight": "balanced"},
        lambda p: RandomForestClassifier(**p, n_jobs=-1, random_state=SEED), None),
    "LightGBM": ("Boosting", {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
                              "is_unbalance": True},
        lambda p: LGBMClassifier(**p, random_state=SEED, verbose=-1), None),
    "MLP": ("Neural network", {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "early_stopping": True,
                               "max_iter": 50},
        lambda p: make_pipeline(StandardScaler(), MLPClassifier(**{**p, "hidden_layer_sizes": tuple(p["hidden_layer_sizes"])},
                                                                random_state=SEED)), 500_000),
}

if __name__ == "__main__":
    train = pd.read_parquet(TRAIN)
    test = pd.read_parquet(TEST)
    print(f"train rows={len(train)} trips={train.trip_id.nunique()} | test rows={len(test)} trips={test.trip_id.nunique()}")
    for name, (family, params, builder, sub) in MODELS.items():
        run_experiment(name, family, lambda: builder(params), params, train, test, train_subsample=sub)
    print(pd.read_csv("results/experiments.csv")[["model", "cv_top1_mean", "cv_macro_f1_mean",
                                                  "test_top1", "test_macro_f1", "test_log_loss"]])
