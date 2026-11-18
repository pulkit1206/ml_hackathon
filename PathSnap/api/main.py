"""PathSnap API (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
    PATHSNAP_EVAL_DIR   folder with the evaluation trips (default: data/evaluator_test)
    PATHSNAP_ARTIFACTS  folder with the trained model + decoder parameters (default: artifacts)

The reference routes stay on this machine: they are read only by score_trip(), after the prediction
has been made, and are never returned to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.pipeline import Matcher, score_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test"))
ART = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
if not EVAL_DIR.is_dir():
    log.warning("Evaluation folder %s not found", EVAL_DIR)

matcher = Matcher(ART)
log.info("Model loaded from %s | features=%s | HMM=%s", ART, matcher.features, matcher.hmm)
app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    return {d.name: d for d in list_trips(EVAL_DIR)}


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    ids = sorted(available_trips())
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    trips_ = available_trips()
    if req.trip_id not in trips_:
        log.warning("POST /predict unknown trip_id=%s", req.trip_id)
        raise HTTPException(404, f"Unknown trip_id {req.trip_id}")
    folder = trips_[req.trip_id]
    log.info("POST /predict trip_id=%s | received", req.trip_id)

    result = matcher.predict_trip(folder)
    log.info("trip_id=%s | candidates=%d | HMM steps=%d breaks=%d | prediction %.2fs", req.trip_id,
             result["n_candidates"], result["hmm_stats"]["steps"], result["hmm_stats"]["breaks"], result["runtime_s"])
    scored = score_trip(folder, result)                     # server-side only
    metrics = scored[0] if scored else None
    if metrics:
        log.info("trip_id=%s | accuracy HMM=%.4f classifier-only=%.4f | continuity HMM=%.3f | scored points=%d",
                 req.trip_id, metrics["accuracy_hmm"], metrics["accuracy_clf"], metrics["continuity_hmm"],
                 metrics["n_scored_points"])

    trip = result["trip"]
    lat, lon = trip["nodes"].lat.to_numpy(), trip["nodes"].lon.to_numpy()
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    seq = [int(a) for a in result["pred_hmm"] if a >= 0]
    ordered = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]   # drive order, repeats collapsed
    segments = [[[float(lat[u[a]]), float(lon[u[a]])], [float(lat[v[a]]), float(lon[v[a]])]] for a in ordered]
    tr = trip["track"]
    step = max(1, len(tr) // 2000)
    out = {
        "trip_id": req.trip_id,
        "gps": tr[["lat", "lon"]].to_numpy()[::step].round(7).tolist(),
        "predicted_route_segments": segments,
        "predicted_arc_per_point": result["pred_hmm"].tolist(),
        "metrics": metrics,
        "hmm_chain_breaks": result["hmm_stats"]["breaks"],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    log.info("POST /predict trip_id=%s | done in %.2fs", req.trip_id, out["runtime_s"])
    return out
