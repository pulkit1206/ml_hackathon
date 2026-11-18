"""Build the candidate dataset (one row per GPS point x candidate arc) for train and student_test.

Labels: team label file (notebooks/02_Route_Label_Generation.ipynb) where available, otherwise the same
monotonic alignment re-implemented in pathsnap/labels.py (trips with route continuity breaks).
GPS points whose reference arc is > 100 m away (is_matched = False) are excluded: their label is unreliable.

    python scripts/build_dataset.py
"""
import time

import numpy as np
import pandas as pd

from common import MODELS, TEAM_LABELS, TEST_DIR, TEST_PQ, TRAIN_DIR, TRAIN_PQ
from pathsnap.features import RADIUS_M, candidate_table, prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import TeamLabels, reference_labels

team = TeamLabels(TEAM_LABELS)
summary = []
for part, folder, out in (("train", TRAIN_DIR, TRAIN_PQ), ("student_test", TEST_DIR, TEST_PQ)):
    parts = []
    for tdir in list_trips(folder):
        t0 = time.perf_counter()
        trip = prepare_trip(load_trip(tdir))
        ref, matched, source = reference_labels(trip, team)
        cand = candidate_table(trip)
        cand = cand[matched[cand.point_idx.to_numpy()]].copy()          # keep reliably-labelled points only
        cand["label"] = (cand.arc_id.to_numpy() == ref[cand.point_idx.to_numpy()]).astype(np.int8)
        hit = cand.groupby("point_idx").label.max()
        n_lab = int(matched.sum())
        summary.append({"partition": part, "trip_id": trip["trip_id"], "gps_points": len(ref),
                        "labelled_points": n_lab, "points_with_candidates": int(len(hit)),
                        "candidate_rows": len(cand), "positive_share": cand.label.mean() if len(cand) else np.nan,
                        "candidate_miss_rate": 1 - hit.sum() / max(n_lab, 1), "label_source": source,
                        "seconds": round(time.perf_counter() - t0, 2)})
        parts.append(cand)
        s = summary[-1]
        print(f"{part} {s['trip_id']}: {s['gps_points']} pts, {s['candidate_rows']} rows, "
              f"miss={s['candidate_miss_rate']:.3f}, labels={source}")
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out, index=False)
    sm = pd.DataFrame([s for s in summary if s["partition"] == part])
    print(f"\n== {part}: trips={len(sm)} rows={len(df):,} positive share={df.label.mean():.3f} "
          f"candidate miss rate={1 - (sm.labelled_points * (1 - sm.candidate_miss_rate)).sum() / sm.labelled_points.sum():.4f} "
          f"(radius {RADIUS_M:.0f} m)\n")
pd.DataFrame(summary).to_csv(MODELS / "dataset_summary.csv", index=False)
