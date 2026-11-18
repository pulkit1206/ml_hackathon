import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.candidates.dataset import MODEL_FEATURE_COLUMNS
from src.candidates.features import build_candidate_features
from src.candidates.generator import generate_candidates, load_track
from src.candidates.geometry import load_road_network

DEFAULT_MODEL_PATH = Path("outputs/knn/knn_k25_pipeline.joblib")

PREDICTION_COLUMNS = [
    "trip_id",
    "point_id",
    "candidate_arc_id",
    "distance_m",
    "distance_rank",
    "predicted_probability",
    "predicted_label",
    "prediction_rank",
    "is_top1",
]


def load_knn_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load a trained Distance-Weighted KNN sklearn Pipeline artifact without retraining.
    """
    return joblib.load(model_path)


def _empty_predictions_df():
    return pd.DataFrame({
        "trip_id": pd.Series(dtype="object"),
        "point_id": pd.Series(dtype="int64"),
        "candidate_arc_id": pd.Series(dtype="int64"),
        "distance_m": pd.Series(dtype="float64"),
        "distance_rank": pd.Series(dtype="int64"),
        "predicted_probability": pd.Series(dtype="float64"),
        "predicted_label": pd.Series(dtype="int64"),
        "prediction_rank": pd.Series(dtype="int64"),
        "is_top1": pd.Series(dtype="bool"),
    })


def predict_candidates(
    features_df,
    model=None,
    model_path=DEFAULT_MODEL_PATH,
    threshold=0.5,
):
    """
    Score candidate-level features using the saved KNN pipeline and return
    ranked candidate predictions per (trip_id, point_id).

    Does not require or use any ground-truth labels.
    Deterministic tie-breaking order:
        1. Highest predicted_probability
        2. Smallest distance_m
        3. Smallest distance_rank
        4. Smallest candidate_arc_id
    """
    if features_df.empty:
        return _empty_predictions_df()

    missing_cols = [c for c in MODEL_FEATURE_COLUMNS if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required model feature columns: {missing_cols}")

    if model is None:
        model = load_knn_model(model_path)

    df = features_df.copy()
    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    df["point_id"] = df["point_id"].astype("int64")
    df["candidate_arc_id"] = df["candidate_arc_id"].astype("int64")

    X = df[MODEL_FEATURE_COLUMNS].astype("float64")
    probs = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "trip_id": df["trip_id"],
        "point_id": df["point_id"],
        "candidate_arc_id": df["candidate_arc_id"],
        "distance_m": df["distance_m"].astype("float64"),
        "distance_rank": df["distance_rank"].astype("int64"),
        "predicted_probability": probs.astype("float64"),
        "predicted_label": (probs >= threshold).astype("int64"),
    })

    out = out.sort_values(
        by=["trip_id", "point_id", "predicted_probability", "distance_m", "distance_rank", "candidate_arc_id"],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    out["prediction_rank"] = (out.groupby(["trip_id", "point_id"], sort=False).cumcount() + 1).astype("int64")
    out["is_top1"] = out["prediction_rank"] == 1
    return out[PREDICTION_COLUMNS]


def select_top1_predictions(predictions_df):
    """
    Return the single highest-scoring candidate per (trip_id, point_id).
    """
    if predictions_df.empty:
        return _empty_predictions_df()
    return predictions_df.loc[predictions_df["is_top1"]].reset_index(drop=True)


def predict_trip(
    trip_folder,
    radius_m=50,
    model=None,
    model_path=DEFAULT_MODEL_PATH,
    threshold=0.5,
):
    """
    Run end-to-end candidate generation, feature extraction, and KNN scoring
    for a single trip folder (using only .nodes, .arcs, and .track).
    """
    trip_path = Path(trip_folder)
    trip_id = trip_path.name

    t0 = time.perf_counter()
    network = load_road_network(trip_path / f"{trip_id}.nodes", trip_path / f"{trip_id}.arcs")
    track_df = load_track(trip_path / f"{trip_id}.track")
    candidates_df = generate_candidates(network=network, track_df=track_df, trip_id=trip_id, radius_m=radius_m)
    features_df = build_candidate_features(network, track_df, candidates_df)
    prep_time_s = time.perf_counter() - t0

    if model is None:
        model = load_knn_model(model_path)

    t1 = time.perf_counter()
    ranked_df = predict_candidates(features_df, model=model, threshold=threshold)
    infer_time_s = time.perf_counter() - t1

    top1_df = select_top1_predictions(ranked_df)
    timing = {
        "trip_id": trip_id,
        "track_observations": int(len(track_df)),
        "candidate_rows": int(len(ranked_df)),
        "observations_with_predictions": int(len(top1_df)),
        "prep_time_s": float(prep_time_s),
        "knn_inference_time_s": float(infer_time_s),
    }
    return ranked_df, top1_df, timing


def run_student_test_inference(
    student_test_dir="Student folder/student_test",
    output_dir="outputs/knn",
    model_path=DEFAULT_MODEL_PATH,
    radius_m=50,
):
    """
    Run inference on the permitted 10 student_test trips using the saved k=25 pipeline.
    """
    test_root = Path(student_test_dir)
    trip_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    model = load_knn_model(model_path)

    all_ranked = []
    all_top1 = []
    timings = []

    for trip_dir in trip_dirs:
        ranked_df, top1_df, timing = predict_trip(trip_dir, radius_m=radius_m, model=model)
        all_ranked.append(ranked_df)
        all_top1.append(top1_df)
        timings.append(timing)

    ranked_all = pd.concat(all_ranked, ignore_index=True) if all_ranked else _empty_predictions_df()
    top1_all = pd.concat(all_top1, ignore_index=True) if all_top1 else _empty_predictions_df()
    timings_df = pd.DataFrame(timings)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ranked_all.to_csv(out_path / "student_test_predictions_k25.csv", index=False)
    top1_all.to_csv(out_path / "student_test_top1_k25.csv", index=False)
    timings_df.to_csv(out_path / "student_test_inference_timings.csv", index=False)

    return ranked_all, top1_all, timings_df


if __name__ == "__main__":
    ranked, top1, timings = run_student_test_inference()
    print("Student-test inference timings (10 permitted trips, saved k=25 model):")
    print(timings.to_string(index=False))
    print("\nTotal candidate rows scored:", len(ranked))
    print("Total Top-1 observations selected:", len(top1))
    print("Total KNN inference time (s):", round(float(timings["knn_inference_time_s"].sum()), 4))
