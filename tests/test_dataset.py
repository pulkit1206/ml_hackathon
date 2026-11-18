import pandas as pd

from src.candidates.dataset import (
    DATASET_COLUMNS,
    EXCLUDED_GROUND_TRUTH_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    clean_reference_labels,
    join_features_with_labels,
)


def test_clean_reference_labels_rules(tmp_path):
    csv_path = tmp_path / "labels.csv"
    raw = pd.DataFrame({
        "trip_id": ["00000001", "00000001", "00000054", "00000054", "00000054"],
        "gps_row_index": [0, 1, 100, 2704, 19312],
        "arc_id": [10, 20, 30, 40, 50],  # Non-null arc_id even when unmatched!
        "route_position": [0, 1, 2, 3, 4],
        "is_matched": [True, False, True, True, True],
    })
    raw.to_csv(csv_path, index=False)

    cleaned, audit = clean_reference_labels(csv_path)
    assert audit["total_raw_label_rows"] == 5
    assert audit["raw_unmatched_rows"] == 1
    assert audit["trip_00000054_excluded_rows"] == 2
    assert audit["cleaned_matched_rows"] == 2

    assert list(cleaned["trip_id"]) == ["00000001", "00000054"]
    assert list(cleaned["gps_row_index"]) == [0, 100]
    assert (cleaned["trip_id"].str.len() == 8).all()


def test_join_features_with_labels_and_leakage_exclusion():
    features = pd.DataFrame({
        "trip_id": ["00000001", "00000001", "00000001"],
        "point_id": [0, 0, 1],
        "candidate_arc_id": [10, 11, 12],
        "distance_m": [1.0, 15.0, 2.0],
        "distance_rank": [1, 2, 1],
        "arc_length_m": [100.0, 80.0, 90.0],
        "along_arc_fraction": [0.2, 0.5, 0.3],
        "candidate_count": [2, 2, 1],
        "distance_gap_m": [0.0, 14.0, 0.0],
        "gps_heading_deg": [np.nan, np.nan, 10.0] if False else [float("nan"), float("nan"), 10.0],
        "road_heading_deg": [0.0, 180.0, 12.0],
        "heading_diff_deg": [float("nan"), float("nan"), 2.0],
        "speed_mps": [float("nan"), float("nan"), 12.0],
        "connected_to_prev": [False, False, True],
    })
    labels = pd.DataFrame({
        "trip_id": ["00000001", "00000001"],
        "gps_row_index": [0, 1],
        "arc_id": [10, 12],
        "route_position": [0, 1],
        "is_matched": [True, True],
    })

    ds, stats = join_features_with_labels(features, labels)
    assert list(ds.columns) == DATASET_COLUMNS
    assert "heading_diff_deg" in MODEL_FEATURE_COLUMNS
    assert not (set(ds.columns) & EXCLUDED_GROUND_TRUTH_COLUMNS)
    assert list(ds["label"]) == [1, 0, 1]
    assert stats["positive_rows"] == 2
    assert stats["negative_rows"] == 1
    assert stats["observations_with_positive_candidate"] == 2
