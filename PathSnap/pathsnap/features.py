"""Candidate generation and feature engineering.

One row per (GPS point, candidate arc within RADIUS_M metres). No feature uses the route,
so exactly the same function is used for training and for inference on unseen trips.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .geo import angle_diff, bearing, project_trip

RADIUS_M = 50.0

FEATURES_ALL = ["dist", "dist_rank", "dist_minus_min", "n_cands", "heading_diff",
                "speed", "time_gap", "arc_len", "proj_frac", "conn_prev_nearest"]


def arc_geometry(trip):
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    nx, ny = trip["nx"], trip["ny"]
    coords = np.stack([np.stack([nx[u], ny[u]], 1), np.stack([nx[v], ny[v]], 1)], 1)
    trip["arc_geoms"] = shapely.linestrings(coords)
    trip["arc_len"] = np.hypot(nx[v] - nx[u], ny[v] - ny[u])
    trip["arc_bear"] = bearing(nx[v] - nx[u], ny[v] - ny[u])
    trip["tree"] = STRtree(trip["arc_geoms"])
    return trip


def motion(trip):
    """Heading and speed from neighbouring points (centred difference), time since previous point."""
    px, py, t = trip["px"], trip["py"], trip["track"].t.to_numpy(float)
    n = len(px)
    i = np.arange(n)
    a, b = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    dx, dy = px[b] - px[a], py[b] - py[a]
    dt = np.maximum(t[b] - t[a], 1e-6)
    trip["gps_bear"] = bearing(dx, dy)
    trip["speed"] = np.hypot(dx, dy) / dt
    trip["time_gap"] = np.diff(t, prepend=t[0])
    return trip


def prepare_trip(trip):
    """Projection + arc geometry + motion. Call once per trip."""
    return motion(arc_geometry(project_trip(trip)))


def candidate_table(trip, radius=RADIUS_M):
    """Vectorised candidate search + features for every GPS point of a prepared trip."""
    pts = shapely.points(trip["px"], trip["py"])
    pi, ai = trip["tree"].query(pts, predicate="dwithin", distance=radius)
    order = np.lexsort((ai, pi))
    pi, ai = pi[order], ai[order]
    g = trip["arc_geoms"][ai]
    dist = shapely.distance(g, pts[pi])
    frac = shapely.line_locate_point(g, pts[pi], normalized=True)

    df = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist": dist, "proj_frac": frac})
    grp = df.groupby("point_idx")
    df["dist_rank"] = grp.dist.rank(method="first").astype(int) - 1
    df["dist_minus_min"] = df.dist - grp.dist.transform("min")
    df["n_cands"] = grp.dist.transform("size")
    df["heading_diff"] = angle_diff(trip["gps_bear"][pi], trip["arc_bear"][ai])
    df["speed"] = trip["speed"][pi]
    df["time_gap"] = trip["time_gap"][pi]
    df["arc_len"] = trip["arc_len"][ai]

    # is this arc the same as / sharing a node with the previous point's nearest arc?
    nearest = df[df.dist_rank == 0].set_index("point_idx").arc_id
    prev_near = nearest.reindex(df.point_idx - 1).to_numpy()
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    ok = ~np.isnan(prev_near)
    pn = np.where(ok, prev_near, 0).astype(np.int64)
    a = df.arc_id.to_numpy()
    shared = (a == pn) | (u[a] == u[pn]) | (u[a] == v[pn]) | (v[a] == u[pn]) | (v[a] == v[pn])
    df["conn_prev_nearest"] = (ok & shared).astype(int)
    df.insert(0, "trip_id", trip["trip_id"])
    return df
