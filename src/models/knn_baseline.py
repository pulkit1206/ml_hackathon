import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.candidates.dataset import (
    EXCLUDED_GROUND_TRUTH_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)

DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_ROWS = 200_000
DEFAULT_N_SPLITS = 5


def clean_candidates_for_training(df):
    """
    Validate and clean candidate dataset for supervised KNN training:
    - Preserve trip_id as 8-character string.
    - Exclude GPS observations without at least one positive candidate.
    - Report candidate-generation misses separately.
    - Verify no infinite values and no ground-truth-derived columns in X.
    """
    data = df.copy()
    data["trip_id"] = data["trip_id"].astype(str).str.zfill(8)
    if not (data["trip_id"].str.len() == 8).all():
        raise ValueError("All trip_id values must be 8-character strings.")

    leaked = set(MODEL_FEATURE_COLUMNS) & EXCLUDED_GROUND_TRUTH_COLUMNS
    if leaked:
        raise AssertionError(f"Ground-truth-derived columns found in features: {leaked}")

    numeric_feats = [c for c in MODEL_FEATURE_COLUMNS if c != "connected_to_prev"]
    if np.isinf(data[numeric_feats].to_numpy(dtype="float64")).any():
        raise ValueError("Infinite values detected in model features.")

    nan_counts = {col: int(data[col].isna().sum()) for col in MODEL_FEATURE_COLUMNS if data[col].isna().any()}

    obs_max_label = data.groupby(["trip_id", "point_id"], sort=False)["label"].transform("max")
    miss_mask = obs_max_label == 0

    total_obs = int(data.groupby(["trip_id", "point_id"], sort=False).ngroups)
    miss_df = data.loc[miss_mask]
    miss_obs = int(miss_df.groupby(["trip_id", "point_id"], sort=False).ngroups)
    miss_rows = int(len(miss_df))

    cleaned = data.loc[~miss_mask].reset_index(drop=True)
    kept_obs = int(cleaned.groupby(["trip_id", "point_id"], sort=False).ngroups)

    stats = {
        "input_candidate_rows": int(len(data)),
        "input_observations": total_obs,
        "candidate_miss_observations_without_positive": miss_obs,
        "candidate_miss_rows_excluded": miss_rows,
        "cleaned_candidate_rows": int(len(cleaned)),
        "cleaned_observations": kept_obs,
        "cleaned_positive_rows": int((cleaned["label"] == 1).sum()),
        "cleaned_negative_rows": int((cleaned["label"] == 0).sum()),
        "feature_nan_counts_before_imputation": nan_counts,
    }
    return cleaned, stats


def sample_balanced_by_trip(cleaned_df, target_rows=DEFAULT_SAMPLE_ROWS, random_seed=DEFAULT_RANDOM_SEED):
    """
    Reproducibly sample whole (trip_id, point_id) observations so that:
    - Total candidate rows are approximately target_rows.
    - Every sampled observation retains its positive candidate and all its negative candidates.
    - Large trips are capped via water-filling so a few large trips do not dominate.
    """
    if len(cleaned_df) <= target_rows:
        sampled = cleaned_df.copy().reset_index(drop=True)
        return sampled, {
            "random_seed": int(random_seed),
            "target_rows": int(target_rows),
            "sampled_rows": int(len(sampled)),
            "sampled_trips": int(sampled["trip_id"].nunique()),
            "sampled_observations": int(sampled.groupby(["trip_id", "point_id"]).ngroups),
            "sampled_positive_rows": int((sampled["label"] == 1).sum()),
            "sampled_negative_rows": int((sampled["label"] == 0).sum()),
            "sampled_positive_rate": float((sampled["label"] == 1).mean()) if len(sampled) else 0.0,
        }

    rng = np.random.default_rng(random_seed)

    # Precompute observation row counts per trip.
    obs_sizes = (
        cleaned_df.groupby(["trip_id", "point_id"], sort=False)
        .size()
        .reset_index(name="n_rows")
    )
    trip_totals = obs_sizes.groupby("trip_id")["n_rows"].sum().sort_values()
    trips_sorted = trip_totals.index.tolist()

    # Water-filling budget per trip so small trips keep all rows and large trips are capped.
    remaining_budget = int(target_rows)
    remaining_trips = len(trips_sorted)
    trip_caps = {}
    for tid in trips_sorted:
        fair_share = remaining_budget / remaining_trips
        t_total = int(trip_totals[tid])
        if t_total <= fair_share:
            trip_caps[tid] = t_total
            remaining_budget -= t_total
        else:
            trip_caps[tid] = int(round(fair_share))
            remaining_budget -= trip_caps[tid]
        remaining_trips -= 1

    selected_obs_keys = []
    for tid in sorted(trips_sorted):
        cap = trip_caps[tid]
        trip_obs = obs_sizes.loc[obs_sizes["trip_id"] == tid]
        if int(trip_obs["n_rows"].sum()) <= cap:
            selected_obs_keys.append(trip_obs[["trip_id", "point_id"]])
            continue

        perm = rng.permutation(len(trip_obs))
        shuffled = trip_obs.iloc[perm]
        cum_rows = shuffled["n_rows"].cumsum().to_numpy()
        # Include observations up to cap (at least 1 observation).
        cutoff = int(np.searchsorted(cum_rows, cap, side="right"))
        cutoff = max(1, min(cutoff, len(shuffled)))
        selected_obs_keys.append(shuffled.iloc[:cutoff][["trip_id", "point_id"]])

    selected_keys_df = pd.concat(selected_obs_keys, ignore_index=True)
    sampled = cleaned_df.merge(selected_keys_df, on=["trip_id", "point_id"], how="inner", sort=False)
    sampled = sampled.sort_values(["trip_id", "point_id", "distance_rank", "candidate_arc_id"]).reset_index(drop=True)

    stats = {
        "random_seed": int(random_seed),
        "target_rows": int(target_rows),
        "sampled_rows": int(len(sampled)),
        "sampled_trips": int(sampled["trip_id"].nunique()),
        "sampled_observations": int(sampled.groupby(["trip_id", "point_id"]).ngroups),
        "sampled_positive_rows": int((sampled["label"] == 1).sum()),
        "sampled_negative_rows": int((sampled["label"] == 0).sum()),
        "sampled_positive_rate": float((sampled["label"] == 1).mean()) if len(sampled) else 0.0,
    }
    return sampled, stats


def build_knn_pipeline(n_neighbors, weights="distance", metric="euclidean"):
    """
    Build a leak-free scikit-learn Pipeline with median imputation,
    StandardScaler, and distance-weighted KNeighborsClassifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1)),
    ])


def select_top1_candidates(pred_df):
    """
    For each (trip_id, point_id), select the candidate with highest prob_pos.
    Deterministic tie-breaking: smallest distance_m, then distance_rank, then candidate_arc_id.
    """
    ordered = pred_df.sort_values(
        by=["trip_id", "point_id", "prob_pos", "distance_m", "distance_rank", "candidate_arc_id"],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    )
    top1 = ordered.drop_duplicates(subset=["trip_id", "point_id"], keep="first").reset_index(drop=True)
    return top1


def compute_evaluation_metrics(y_true, y_prob_pos, pred_df, threshold=0.5):
    """
    Compute observation-level Top-1 accuracy and candidate-level classification metrics.
    """
    y_pred = (np.asarray(y_prob_pos) >= threshold).astype("int64")
    top1 = select_top1_candidates(pred_df)
    top1_acc = float((top1["label"] == 1).mean()) if len(top1) > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "observation_top1_accuracy": top1_acc,
        "observations_evaluated": int(len(top1)),
        "candidate_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "candidate_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "candidate_log_loss": float(log_loss(y_true, y_prob_pos, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_knn_group_cv(sampled_df, n_neighbors, n_splits=DEFAULT_N_SPLITS):
    """
    Run GroupKFold cross-validation grouped by trip_id for a given k.
    """
    X = sampled_df[MODEL_FEATURE_COLUMNS].astype("float64")
    y = sampled_df["label"].to_numpy(dtype="int64")
    groups = sampled_df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(sampled_df), dtype="float64")
    fold_rows = []
    total_fit_time = 0.0
    total_infer_time = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        # Verify strict trip separation between train and validation folds.
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        if train_trips & val_trips:
            raise AssertionError("Group leakage detected between train and validation folds.")

        pipe = build_knn_pipeline(n_neighbors=n_neighbors, weights="distance", metric="euclidean")

        t0 = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        fit_s = time.perf_counter() - t0
        total_fit_time += fit_s

        t1 = time.perf_counter()
        probs = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        infer_s = time.perf_counter() - t1
        total_infer_time += infer_s

        oof_probs[val_idx] = probs

        val_df = sampled_df.iloc[val_idx][["trip_id", "point_id", "candidate_arc_id", "distance_m", "distance_rank", "label"]].copy()
        val_df["prob_pos"] = probs
        fold_metrics = compute_evaluation_metrics(y[val_idx], probs, val_df)
        fold_rows.append({
            "k": int(n_neighbors),
            "fold": int(fold_idx),
            "val_trips": int(len(val_trips)),
            "val_rows": int(len(val_idx)),
            "fit_time_s": float(fit_s),
            "inference_time_s": float(infer_s),
            "observation_top1_accuracy": fold_metrics["observation_top1_accuracy"],
            "candidate_macro_f1": fold_metrics["candidate_macro_f1"],
            "candidate_precision": fold_metrics["candidate_precision"],
            "candidate_recall": fold_metrics["candidate_recall"],
            "candidate_log_loss": fold_metrics["candidate_log_loss"],
        })

    oof_df = sampled_df[["trip_id", "point_id", "candidate_arc_id", "distance_m", "distance_rank", "label"]].copy()
    oof_df["prob_pos"] = oof_probs
    oof_df["pred_label"] = (oof_probs >= 0.5).astype("int64")

    overall_metrics = compute_evaluation_metrics(y, oof_probs, oof_df)
    overall_metrics["k"] = int(n_neighbors)
    overall_metrics["weights"] = "distance"
    overall_metrics["metric"] = "euclidean"
    overall_metrics["total_fit_time_s"] = float(total_fit_time)
    overall_metrics["total_inference_time_s"] = float(total_infer_time)

    # Fit final pipeline on full sampled dataset for artifact saving.
    final_pipe = build_knn_pipeline(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
    t_final = time.perf_counter()
    final_pipe.fit(X, y)
    overall_metrics["full_sample_fit_time_s"] = float(time.perf_counter() - t_final)

    return overall_metrics, pd.DataFrame(fold_rows), oof_df, final_pipe


def run_knn_baseline_experiment(
    dataset_path="outputs/dataset/knn_train_candidates_r50.csv",
    output_dir="outputs/knn",
    k_values=(5, 25),
    target_rows=DEFAULT_SAMPLE_ROWS,
    n_splits=DEFAULT_N_SPLITS,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """
    Run the end-to-end Distance-Weighted KNN baseline comparison and save all artifacts.
    """
    raw_df = pd.read_csv(dataset_path, dtype={"trip_id": str})
    cleaned_df, cleaning_stats = clean_candidates_for_training(raw_df)
    sampled_df, sampling_stats = sample_balanced_by_trip(
        cleaned_df, target_rows=target_rows, random_seed=random_seed
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_folds = []

    for k in k_values:
        metrics, folds_df, oof_df, final_pipe = evaluate_knn_group_cv(
            sampled_df, n_neighbors=k, n_splits=n_splits
        )
        all_metrics.append(metrics)
        all_folds.append(folds_df)

        joblib.dump(final_pipe, out_path / f"knn_k{k}_pipeline.joblib")
        oof_df.to_csv(out_path / f"oof_predictions_k{k}.csv", index=False)

    cv_folds_df = pd.concat(all_folds, ignore_index=True)
    cv_folds_df.to_csv(out_path / "cv_fold_results.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_path / "evaluation_comparison.csv", index=False)

    config = {
        "dataset_path": str(dataset_path),
        "random_seed": int(random_seed),
        "target_sample_rows": int(target_rows),
        "n_splits": int(n_splits),
        "cv_strategy": "GroupKFold(groups=trip_id)",
        "k_values": list(k_values),
        "weights": "distance",
        "metric": "euclidean",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "cleaning_stats": cleaning_stats,
        "sampling_stats": sampling_stats,
    }
    with open(out_path / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(out_path / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    return config, metrics_df, cv_folds_df


if __name__ == "__main__":
    config, metrics_df, cv_folds_df = run_knn_baseline_experiment()
    print("Cleaning & Sampling Stats:")
    print(json.dumps({"cleaning": config["cleaning_stats"], "sampling": config["sampling_stats"]}, indent=2))
    print("\nBaseline Comparison (GroupKFold, 5 folds):")
    print(metrics_df.to_string(index=False))
