"""Reference labels: which route arc each GPS point belongs to.

Primary source: the team's label file produced by notebooks/02_Route_Label_Generation.ipynb
(monotonic dynamic-programming alignment of GPS points to the ordered .route arcs, 100 m threshold).

The same algorithm is re-implemented here (vectorised) for
  * trips the notebook skipped because their .route has continuity breaks, and
  * server-side scoring inside the API (evaluator trips),
so that every partition is labelled with one consistent method.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_THRESHOLD_M = 100.0
MAX_DP_CELLS = 60_000_000


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    den = vx * vx + vy * vy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - ax) * vx + (py - ay) * vy) / den
    t = np.clip(np.where(den > 0, t, 0.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def align_route(trip):
    """Monotonic alignment: route position never decreases along the (time-ordered) track.
    Minimises the sum of squared point-to-segment distances. Returns (ref_arc, dist_m, matched)."""
    route = trip["route"]
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    ax, ay = trip["nx"][u[route]], trip["ny"][u[route]]
    bx, by = trip["nx"][v[route]], trip["ny"][v[route]]
    n, m = len(trip["px"]), len(route)
    if n * m > MAX_DP_CELLS:
        raise MemoryError(f"{trip['trip_id']}: alignment needs {n*m:,} cells")
    back = np.zeros((n, m), dtype=np.int32)
    idx = np.arange(m)
    prev = None
    for i in range(n):
        cost = _seg_dist(trip["px"][i], trip["py"][i], ax, ay, bx, by) ** 2
        if prev is not None:
            run_min = np.minimum.accumulate(prev)
            # index of the running minimum (latest position that reached it)
            arg = np.maximum.accumulate(np.where(prev == run_min, idx, 0))
            cost = cost + run_min
            back[i] = arg
        prev = cost
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    d = np.array([_seg_dist(trip["px"][i], trip["py"][i], ax[j], ay[j], bx[j], by[j])
                  for i, j in enumerate(pos)])
    return route[pos], d, d <= MATCH_THRESHOLD_M


class TeamLabels:
    """Lookup into the team's label parquet files (may be missing -> fall back to align_route)."""

    def __init__(self, label_files):
        frames = [pd.read_parquet(f) for f in label_files if Path(f).exists()]
        self.df = pd.concat(frames, ignore_index=True) if frames else None
        if self.df is not None:
            self.df["trip_id"] = self.df.trip_id.astype(str).str.zfill(8)

    def get(self, trip_id, n_points):
        if self.df is None:
            return None
        d = self.df[self.df.trip_id == trip_id]
        if d.empty:
            return None
        ref = np.full(n_points, -1, dtype=np.int64)
        matched = np.zeros(n_points, dtype=bool)
        ref[d.gps_row_index.to_numpy()] = d.arc_id.to_numpy()
        matched[d.gps_row_index.to_numpy()] = d.is_matched.to_numpy()
        return ref, matched


def reference_labels(trip, team_labels=None):
    """Returns (ref_arc, matched, source). Points with matched=False are excluded from training/scoring."""
    n = len(trip["px"])
    if team_labels is not None:
        got = team_labels.get(trip["trip_id"], n)
        if got is not None:
            return got[0], got[1], "team_notebook"
    ref, _, matched = align_route(trip)
    return ref, matched, "pipeline_alignment"
