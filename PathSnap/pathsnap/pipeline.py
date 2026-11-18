"""End-to-end inference for one trip: candidates -> features -> classifier -> Viterbi.

predict_trip() never reads the .route file. Scoring against the reference route is a
separate function (score_trip) that only the server calls, after prediction.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .features import candidate_table, prepare_trip
from .graph import RoadGraph, route_continuity
from .hmm import HMMParams, classifier_only, viterbi_decode
from .io import load_trip
from .labels import reference_labels


class Matcher:
    def __init__(self, artifacts_dir="artifacts"):
        a = Path(artifacts_dir)
        self.model = joblib.load(a / "final_model.joblib")
        self.features = json.load(open(a / "feature_columns.json"))
        hp = a / "hmm_params.json"
        self.hmm = HMMParams(**json.load(open(hp))) if hp.exists() else HMMParams()

    def predict_trip(self, trip_folder):
        t0 = time.perf_counter()
        trip = prepare_trip(load_trip(trip_folder, with_route=False))   # route NOT loaded
        cand = candidate_table(trip)
        proba = self.model.predict_proba(cand[self.features])[:, 1] if len(cand) else np.array([])
        graph = RoadGraph(trip)
        pred_clf = classifier_only(trip, cand, proba)
        pred_hmm, stats = viterbi_decode(trip, cand, proba, self.hmm, graph)
        return {"trip": trip, "graph": graph, "pred_hmm": pred_hmm, "pred_clf": pred_clf,
                "n_candidates": int(len(cand)), "hmm_stats": stats,
                "runtime_s": round(time.perf_counter() - t0, 2)}


def score_trip(trip_folder, result):
    """Server-side scoring. Loads the reference route separately and compares point by point."""
    ref_trip = load_trip(trip_folder, with_route=True)
    if ref_trip["route"] is None:
        return None
    trip = result["trip"]
    ref_trip.update({k: trip[k] for k in ("px", "py", "nx", "ny")})
    ref, matched, _ = reference_labels(ref_trip)
    g = result["graph"]
    out = {"n_points": int(len(ref)), "n_scored_points": int(matched.sum())}
    for name in ("hmm", "clf"):
        p = result[f"pred_{name}"]
        out[f"accuracy_{name}"] = float((p[matched] == ref[matched]).mean()) if matched.any() else None
        out[f"continuity_{name}"] = route_continuity(g, p)
    ref_set, pred_set = set(ref[matched].tolist()), set(result["pred_hmm"][result["pred_hmm"] >= 0].tolist())
    out["route_arc_precision"] = len(ref_set & pred_set) / max(len(pred_set), 1)
    out["route_arc_recall"] = len(ref_set & pred_set) / max(len(ref_set), 1)
    return out, ref, matched
