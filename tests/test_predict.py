import numpy as np
import pandas as pd

from src.models.predict import (
    PREDICTION_COLUMNS,
    load_knn_model,
    predict_candidates,
    select_top1_predictions,
)


def test_saved_k25_pipeline_loads_and_predicts():
    model = load_knn_model("outputs/knn/knn_k25_pipeline.joblib")
    assert model.n_features_in_ == 11
    assert model.named_steps["knn"].n_neighbors == 25

    features = pd.DataFrame({
        "trip_id": ["00000000", "00000000", "00000000"],
        "point_id": [0, 0, 1],
        "candidate_arc_id": [64973, 117639, 64973],
        "distance_m": [0.7, 15.1, 0.3],
        "distance_rank": [1, 2, 1],
        "arc_length_m": [293.2, 1020.4, 293.2],
        "along_arc_fraction": [0.11, 0.51, 0.26],
        "candidate_count": [2, 2, 1],
        "distance_gap_m": [0.0, 14.4, 0.0],
        "gps_heading_deg": [np.nan, np.nan, 8.9],
        "road_heading_deg": [7.6, 187.7, 7.6],
        "heading_diff_deg": [np.nan, np.nan, 1.3],
        "speed_mps": [np.nan, np.nan, 43.6],
        "connected_to_prev": [False, False, True],
    })

    ranked = predict_candidates(features, model=model)
    assert list(ranked.columns) == PREDICTION_COLUMNS
    assert len(ranked) == 3
    assert (ranked["trip_id"] == "00000000").all()
    assert set(ranked["candidate_arc_id"]) == {64973, 117639}
    assert ((ranked["predicted_probability"] >= 0.0) & (ranked["predicted_probability"] <= 1.0)).all()
    assert set(ranked["predicted_label"]).issubset({0, 1})

    top1 = select_top1_predictions(ranked)
    assert len(top1) == 2
    assert list(top1["point_id"]) == [0, 1]
    assert top1["is_top1"].all()


def test_predict_candidates_empty_input():
    model = load_knn_model("outputs/knn/knn_k25_pipeline.joblib")
    empty = pd.DataFrame()
    out = predict_candidates(empty, model=model)
    assert out.empty
    assert list(out.columns) == PREDICTION_COLUMNS
