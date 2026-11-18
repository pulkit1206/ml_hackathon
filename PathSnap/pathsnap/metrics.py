"""Evaluation metrics (candidate level and observation level)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def top1_pick(df, proba):
    """Highest-probability candidate for every GPS point."""
    tmp = df[["trip_id", "point_idx", "arc_id", "label"]].copy()
    tmp["p"] = proba
    return tmp.loc[tmp.groupby(["trip_id", "point_idx"]).p.idxmax()]


def top1_accuracy(df, proba):
    """Observation-level Top-1: share of (labelled) GPS points whose best candidate is the reference arc."""
    return float(top1_pick(df, proba).label.mean())


def candidate_metrics(df, proba, threshold=0.5):
    y = df.label.to_numpy()
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_pos": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "top1": top1_accuracy(df, proba),
    }


def candidate_miss_rate(df, n_labelled_points):
    """Share of labelled GPS points whose reference arc is NOT among the candidates
    (includes points with no candidate at all)."""
    hit = df.groupby(["trip_id", "point_idx"]).label.max().sum()
    return 1.0 - hit / max(n_labelled_points, 1)
