import numpy as np
import pandas as pd

from src.candidates.dataset import EXCLUDED_GROUND_TRUTH_COLUMNS, MODEL_FEATURE_COLUMNS
from src.models.knn_baseline import (
    clean_candidates_for_training,
    evaluate_knn_group_cv,
    sample_balanced_by_trip,
    select_top1_candidates,
)


def _make_synthetic_knn_dataset():
    """
    Create a synthetic candidate dataset with 5 trips:
    - 4 trips with valid observations (each having 1 positive and 2 negative candidates)
    - 1 observation in trip 00000001 with NO positive candidate (candidate miss)
    """
    rows = []
    rng = np.random.default_rng(123)
    for t_idx in range(1, 6):
        tid = f"{t_idx:08d}"
        for pid in range(10):
            # Observation (00000001, point_id=0) is a candidate miss (no positive).
            is_miss_obs = tid == "00000001" and pid == 0
            for rank in (1, 2, 3):
                is_pos = (not is_miss_obs) and (rank == 1)
                rows.append({
                    "trip_id": tid,
                    "point_id": pid,
                    "candidate_arc_id": 1000 + pid * 10 + rank,
                    "distance_m": float(rank * 5.0 + rng.uniform(0, 0.5)),
                    "distance_rank": rank,
                    "arc_length_m": 100.0,
                    "along_arc_fraction": 0.5,
                    "candidate_count": 3,
                    "distance_gap_m": float((rank - 1) * 5.0),
                    "gps_heading_deg": np.nan if pid == 0 else 10.0,
                    "road_heading_deg": 10.0 if is_pos else 170.0,
                    "heading_diff_deg": np.nan if pid == 0 else (0.0 if is_pos else 160.0),
                    "speed_mps": np.nan if pid == 0 else 15.0,
                    "connected_to_prev": bool(is_pos and pid > 0),
                    "label": int(is_pos),
                })
    return pd.DataFrame(rows)


def test_feature_selection_and_no_target_leakage():
    assert len(MODEL_FEATURE_COLUMNS) == 11
    assert "heading_diff_deg" in MODEL_FEATURE_COLUMNS
    assert "label" not in MODEL_FEATURE_COLUMNS
    assert not (set(MODEL_FEATURE_COLUMNS) & EXCLUDED_GROUND_TRUTH_COLUMNS)


def test_candidate_miss_handling():
    df = _make_synthetic_knn_dataset()
    cleaned, stats = clean_candidates_for_training(df)

    assert stats["candidate_miss_observations_without_positive"] == 1
    assert stats["candidate_miss_rows_excluded"] == 3
    assert stats["cleaned_observations"] == 49
    assert stats["cleaned_positive_rows"] == 49
    # Every remaining observation must have at least one positive candidate.
    assert (cleaned.groupby(["trip_id", "point_id"])["label"].max() == 1).all()


def test_top1_selection_and_deterministic_tie_breaking():
    pred_df = pd.DataFrame({
        "trip_id": ["00000001", "00000001", "00000001", "00000001"],
        "point_id": [0, 0, 1, 1],
        "candidate_arc_id": [20, 10, 30, 40],
        "distance_m": [5.0, 5.0, 12.0, 3.0],
        "distance_rank": [2, 1, 2, 1],
        "label": [0, 1, 0, 1],
        "prob_pos": [0.8, 0.8, 0.2, 0.9],
    })
    top1 = select_top1_candidates(pred_df)
    assert len(top1) == 2
    # Point 0 tied at prob_pos=0.8 and distance_m=5.0 -> distance_rank=1 (arc_id 10) wins.
    assert top1.loc[top1["point_id"] == 0, "candidate_arc_id"].iloc[0] == 10
    # Point 1 -> highest prob_pos=0.9 (arc_id 40) wins.
    assert top1.loc[top1["point_id"] == 1, "candidate_arc_id"].iloc[0] == 40


def test_group_separation_and_reproducibility():
    df = _make_synthetic_knn_dataset()
    cleaned, _ = clean_candidates_for_training(df)

    s1, stats1 = sample_balanced_by_trip(cleaned, target_rows=90, random_seed=42)
    s2, stats2 = sample_balanced_by_trip(cleaned, target_rows=90, random_seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    assert stats1 == stats2

    m1, folds1, oof1, _ = evaluate_knn_group_cv(cleaned, n_neighbors=5, n_splits=5)
    m2, folds2, oof2, _ = evaluate_knn_group_cv(cleaned, n_neighbors=5, n_splits=5)
    assert m1["observation_top1_accuracy"] == m2["observation_top1_accuracy"]
    assert m1["candidate_log_loss"] == m2["candidate_log_loss"]
    pd.testing.assert_frame_equal(oof1, oof2)
    assert len(folds1) == 5
    # Verify 100% of trips are covered across the 5 folds.
    assert folds1["val_trips"].sum() == cleaned["trip_id"].nunique()
