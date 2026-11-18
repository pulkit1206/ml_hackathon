# Person 2 -> Person 4 (HMM / Viterbi) Integration Interface

## Selected Model
- **Artifact:** `outputs/knn/knn_k25_pipeline.joblib` (`SimpleImputer(median)` + `StandardScaler` + `KNeighborsClassifier(n_neighbors=25, weights="distance", metric="euclidean")`)
- **Prediction Module:** `src/models/predict.py`

## Entry Points for Person 4

### 1. Score Pre-Computed Candidate Features
```python
from src.models.predict import load_knn_model, predict_candidates, select_top1_predictions

model = load_knn_model("outputs/knn/knn_k25_pipeline.joblib")
ranked_df = predict_candidates(features_df, model=model)
top1_df = select_top1_predictions(ranked_df)
```

### 2. Score a Trip Folder Directly (`.nodes`, `.arcs`, `.track` only)
```python
from src.models.predict import predict_trip

ranked_df, top1_df, timing = predict_trip("Student folder/student_test/00000000", radius_m=50)
```

## Output Schema (`ranked_df`)
| Column | Dtype | Notes for HMM / Viterbi |
| :--- | :--- | :--- |
| `trip_id` | `str` | 8-character zero-padded trip ID |
| `point_id` | `int64` | Original 0-based GPS observation index |
| `candidate_arc_id` | `int64` | Original 0-based `.arcs` row index |
| `distance_m` | `float64` | Projected point-to-arc distance in metres |
| `distance_rank` | `int64` | 1-based geometric distance rank within `(trip_id, point_id)` |
| `predicted_probability` | `float64` | KNN positive-class probability in `[0.0, 1.0]` (use as emission score / `log(p + eps)`) |
| `predicted_label` | `int64` | Binary label (`predicted_probability >= 0.5`) |
| `prediction_rank` | `int64` | 1-based rank within `(trip_id, point_id)` sorted by `(-predicted_probability, distance_m, distance_rank, candidate_arc_id)` |
| `is_top1` | `bool` | `True` for `prediction_rank == 1` |
