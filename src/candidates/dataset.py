from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.candidates.alignment import load_route, validate_route_connectivity
from src.candidates.features import build_candidate_features
from src.candidates.generator import generate_candidates, load_track
from src.candidates.geometry import load_road_network

ID_COLUMNS = ["trip_id", "point_id", "candidate_arc_id"]

MODEL_FEATURE_COLUMNS = [
    "distance_m",
    "distance_rank",
    "arc_length_m",
    "along_arc_fraction",
    "candidate_count",
    "distance_gap_m",
    "gps_heading_deg",
    "road_heading_deg",
    "heading_diff_deg",
    "speed_mps",
    "connected_to_prev",
]

TARGET_COLUMN = "label"

DATASET_COLUMNS = ID_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]

EXCLUDED_GROUND_TRUTH_COLUMNS = {
    "arc_id",
    "route_position",
    "from_node_id",
    "to_node_id",
    "fraction_along_arc",
    "distance_to_assigned_arc_m",
    "is_matched",
    "label_status",
    "projection",
    "longitude",
    "latitude",
    "timestamp",
    "gps_valid",
    "partition",
}


def clean_reference_labels(labels_path):
    """
    Load train_gps_route_labels.csv read-only and apply validated cleaning rules:
    1. Preserve trip_id as 8-character zero-padded strings.
    2. Exclude Trip 00000054 GPS rows 2704..19312 from supervised training.
    3. Filter unmatched observations using is_matched == True (not arc_id notna).
    """
    raw = pd.read_csv(labels_path, dtype={"trip_id": str})
    raw["trip_id"] = raw["trip_id"].str.zfill(8)

    if not (raw["trip_id"].str.len() == 8).all():
        raise ValueError("All trip_id values must be 8-character strings.")

    total_raw_rows = len(raw)
    raw_trips = sorted(raw["trip_id"].unique().tolist())
    unmatched_total = int((~raw["is_matched"]).sum())

    # Rule 1: Exclude Trip 00000054 GPS rows 2704..19312 inclusive.
    mask_t54_corrupt = (
        (raw["trip_id"] == "00000054")
        & (raw["gps_row_index"] >= 2704)
        & (raw["gps_row_index"] <= 19312)
    )
    t54_excluded_count = int(mask_t54_corrupt.sum())
    after_t54 = raw.loc[~mask_t54_corrupt].copy()

    # Rule 2: Filter unmatched observations using is_matched.
    cleaned = after_t54.loc[after_t54["is_matched"] == True].copy()  # noqa: E712
    cleaned = cleaned[["trip_id", "gps_row_index", "arc_id", "route_position", "is_matched"]].copy()
    cleaned["gps_row_index"] = cleaned["gps_row_index"].astype("int64")
    cleaned["arc_id"] = cleaned["arc_id"].astype("int64")
    cleaned["route_position"] = cleaned["route_position"].astype("int64")

    cleaned_trips = sorted(cleaned["trip_id"].unique().tolist())

    audit = {
        "total_raw_label_rows": int(total_raw_rows),
        "raw_labelled_trip_count": len(raw_trips),
        "raw_unmatched_rows": unmatched_total,
        "trip_00000054_excluded_rows": t54_excluded_count,
        "cleaned_matched_rows": int(len(cleaned)),
        "cleaned_trip_count": len(cleaned_trips),
    }
    return cleaned, audit


def investigate_unlabelled_trips(train_dir, labelled_trip_ids):
    """
    Inspect training trips absent from the reference label file without fabricating labels
    or silently repairing broken routes.
    """
    train_path = Path(train_dir)
    all_trips = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    unlabelled_trips = sorted(set(all_trips) - set(labelled_trip_ids))

    rows = []
    for tid in unlabelled_trips:
        trip_path = train_path / tid
        network = load_road_network(trip_path / f"{tid}.nodes", trip_path / f"{tid}.arcs")
        track_df = load_track(trip_path / f"{tid}.track")
        route_arcs = load_route(trip_path / f"{tid}.route")
        route_val = validate_route_connectivity(network, route_arcs)
        rows.append({
            "trip_id": tid,
            "track_observations": int(len(track_df)),
            "route_length": route_val["route_length"],
            "unique_route_arcs": route_val["unique_route_arcs"],
            "directed_breaks": route_val["directed_breaks"],
            "is_continuous_directed_path": route_val["is_continuous_directed_path"],
        })

    return pd.DataFrame(rows)


def join_features_with_labels(features_df, trip_labels_df):
    """
    Join candidate-level features with cleaned observation-level reference labels
    using (trip_id, point_id) == (trip_id, gps_row_index).

    Generates binary target label = (candidate_arc_id == reference arc_id),
    verifies positive alignment, and excludes ground-truth-derived columns from X.
    """
    if features_df.empty or trip_labels_df.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS), {
            "labelled_observations": int(len(trip_labels_df)),
            "observations_with_candidates": 0,
            "observations_with_positive_candidate": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        }

    labels_subset = trip_labels_df[["trip_id", "gps_row_index", "arc_id"]].rename(
        columns={"gps_row_index": "point_id", "arc_id": "_ref_arc_id"}
    )

    merged = features_df.merge(
        labels_subset,
        on=["trip_id", "point_id"],
        how="inner",
        sort=False,
    )

    if merged.empty:
        return pd.DataFrame(columns=DATASET_COLUMNS), {
            "labelled_observations": int(len(trip_labels_df)),
            "observations_with_candidates": 0,
            "observations_with_positive_candidate": 0,
            "positive_rows": 0,
            "negative_rows": 0,
        }

    merged[TARGET_COLUMN] = (merged["candidate_arc_id"] == merged["_ref_arc_id"]).astype("int64")

    # Verify every positive candidate corresponds to the assigned reference arc.
    positives = merged.loc[merged[TARGET_COLUMN] == 1]
    if not (positives["candidate_arc_id"] == positives["_ref_arc_id"]).all():
        raise AssertionError("Positive candidate does not match assigned reference arc_id.")

    # Drop _ref_arc_id and ensure no ground-truth-derived columns remain.
    merged = merged.drop(columns=["_ref_arc_id"])
    leaked_cols = set(merged.columns) & EXCLUDED_GROUND_TRUTH_COLUMNS
    if leaked_cols:
        raise AssertionError(f"Ground-truth-derived columns leaked into dataset: {leaked_cols}")

    obs_with_cands = int(merged["point_id"].nunique())
    obs_with_pos = int(positives["point_id"].nunique())
    pos_rows = int(len(positives))
    neg_rows = int(len(merged) - pos_rows)

    dataset_df = merged[DATASET_COLUMNS].reset_index(drop=True)
    stats = {
        "labelled_observations": int(len(trip_labels_df)),
        "observations_with_candidates": obs_with_cands,
        "observations_with_positive_candidate": obs_with_pos,
        "positive_rows": pos_rows,
        "negative_rows": neg_rows,
    }
    return dataset_df, stats


def _process_single_trip(task):
    trip_id, trip_dir_str, trip_labels_df, radius_m = task
    trip_path = Path(trip_dir_str)
    network = load_road_network(trip_path / f"{trip_id}.nodes", trip_path / f"{trip_id}.arcs")
    track_df = load_track(trip_path / f"{trip_id}.track")
    candidates_df = generate_candidates(
        network=network,
        track_df=track_df,
        trip_id=trip_id,
        radius_m=radius_m,
    )
    features_df = build_candidate_features(network, track_df, candidates_df)
    dataset_df, stats = join_features_with_labels(features_df, trip_labels_df)
    stats["trip_id"] = trip_id
    return dataset_df, stats


def prepare_knn_dataset(
    labels_path="train_gps_route_labels.csv",
    train_dir="Student folder/train",
    radius_m=50,
    output_dir="outputs/dataset",
    max_workers=4,
):
    """
    Build the cleaned, candidate-level KNN training dataset across all 70 validated trips
    and investigate the 10 unlabelled trips separately.
    """
    cleaned_labels, label_audit = clean_reference_labels(labels_path)
    labelled_trips = sorted(cleaned_labels["trip_id"].unique().tolist())

    unlabelled_df = investigate_unlabelled_trips(train_dir, labelled_trips)

    train_path = Path(train_dir)
    tasks = [
        (
            tid,
            str(train_path / tid),
            cleaned_labels.loc[cleaned_labels["trip_id"] == tid].copy(),
            radius_m,
        )
        for tid in labelled_trips
    ]

    trip_dfs = []
    trip_stats = []
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for df_trip, st in executor.map(_process_single_trip, tasks):
                trip_dfs.append(df_trip)
                trip_stats.append(st)
    else:
        for t in tasks:
            df_trip, st = _process_single_trip(t)
            trip_dfs.append(df_trip)
            trip_stats.append(st)

    dataset_df = pd.concat(trip_dfs, ignore_index=True) if trip_dfs else pd.DataFrame(columns=DATASET_COLUMNS)
    trip_stats_df = pd.DataFrame(trip_stats)

    total_rows = int(len(dataset_df))
    pos_rows = int((dataset_df[TARGET_COLUMN] == 1).sum())
    neg_rows = int((dataset_df[TARGET_COLUMN] == 0).sum())
    total_labelled_obs = int(trip_stats_df["labelled_observations"].sum())
    obs_with_cands = int(trip_stats_df["observations_with_candidates"].sum())
    obs_with_pos = int(trip_stats_df["observations_with_positive_candidate"].sum())

    summary = {
        **label_audit,
        "radius_m": radius_m,
        "unlabelled_trip_count": int(len(unlabelled_df)),
        "total_candidate_rows": total_rows,
        "supports_200k_rows": bool(total_rows >= 200_000),
        "positive_rows": pos_rows,
        "negative_rows": neg_rows,
        "positive_rate": float(pos_rows / total_rows) if total_rows > 0 else 0.0,
        "labelled_observations_total": total_labelled_obs,
        "observations_with_candidates": obs_with_cands,
        "candidate_coverage_rate": float(obs_with_cands / total_labelled_obs) if total_labelled_obs > 0 else 0.0,
        "observations_with_positive_candidate": obs_with_pos,
        "positive_recall_within_radius": float(obs_with_pos / total_labelled_obs) if total_labelled_obs > 0 else 0.0,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(out_path / f"knn_train_candidates_r{radius_m}.csv", index=False)
        trip_stats_df.to_csv(out_path / f"knn_trip_coverage_r{radius_m}.csv", index=False)
        unlabelled_df.to_csv(out_path / "unlabelled_trips_investigation.csv", index=False)
        with open(out_path / f"knn_dataset_validation_summary_r{radius_m}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return dataset_df, summary, unlabelled_df, trip_stats_df


if __name__ == "__main__":
    dataset, summary, unlabelled, _ = prepare_knn_dataset(
        labels_path="train_gps_route_labels.csv",
        train_dir="Student folder/train",
        radius_m=50,
        output_dir="outputs/dataset",
        max_workers=4,
    )
    print("KNN Dataset Preparation Summary:")
    print(json.dumps(summary, indent=2))
    print("\nUnlabelled trips investigation:")
    print(unlabelled.to_string(index=False))
