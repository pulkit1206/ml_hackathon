"""
Build the candidate dataset: one row per (GPS point, candidate arc).

Usage:
    python build_dataset.py --data data/train        --out results/train_candidates.parquet
    python build_dataset.py --data data/student_test --out results/student_test_candidates.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import STRtree, LineString, Point

RADIUS_M = 50          # candidate search radius in metres (tune this: 25/50/75/100)
ROUTE_WINDOW = 6       # how far ahead along the route the label alignment may look

# lon/lat (WGS84) -> metres (UTM zone 36N). Degrees are not a distance unit.
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)


def load_trip(folder):
    tid = os.path.basename(folder.rstrip("/"))
    track = pd.read_csv(f"{folder}/{tid}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(f"{folder}/{tid}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(f"{folder}/{tid}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route_path = f"{folder}/{tid}.route"
    route = pd.read_csv(route_path, header=None)[0].to_numpy() if os.path.exists(route_path) else None
    return tid, track, nodes, arcs, route


def bearing(dx, dy):
    """Compass-style angle in degrees, 0..360."""
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


def angle_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def align_route(px, py, geoms, route):
    """Give each GPS point a reference arc from the route.
    Walk forward through the route only (never backwards), picking the nearest
    route arc within a small window ahead of the current position."""
    ref, pos = [], 0
    for x, y in zip(px, py):
        p = Point(x, y)
        window = range(pos, min(pos + ROUTE_WINDOW, len(route)))
        dists = [geoms[route[k]].distance(p) for k in window]
        best = window[int(np.argmin(dists))]
        pos = best
        ref.append(route[best])
    return np.array(ref)


def build_trip(folder):
    tid, track, nodes, arcs, route = load_trip(folder)

    # project to metres
    nx, ny = to_utm.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    px, py = to_utm.transform(track.lon.to_numpy(), track.lat.to_numpy())

    # one LineString per arc (row index = arc_id)
    u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
    geoms = [LineString([(nx[a], ny[a]), (nx[b], ny[b])]) for a, b in zip(u, v)]
    arc_len = np.hypot(nx[v] - nx[u], ny[v] - ny[u])
    arc_bear = bearing(nx[v] - nx[u], ny[v] - ny[u])
    tree = STRtree(geoms)

    # vehicle heading & speed from neighbouring points (centred difference)
    n = len(px)
    prev_i = np.maximum(np.arange(n) - 1, 0)
    next_i = np.minimum(np.arange(n) + 1, n - 1)
    dx, dy = px[next_i] - px[prev_i], py[next_i] - py[prev_i]
    dt = np.maximum(track.t.to_numpy()[next_i] - track.t.to_numpy()[prev_i], 1e-6)
    gps_bear = bearing(dx, dy)
    speed = np.hypot(dx, dy) / dt
    gap = np.diff(track.t.to_numpy(), prepend=track.t.iloc[0])  # time since previous point

    ref = align_route(px, py, geoms, route) if route is not None else np.full(n, -1)

    rows = []
    prev_nearest = -1
    for i in range(n):
        p = Point(px[i], py[i])
        cand = tree.query(p, predicate="dwithin", distance=RADIUS_M)
        if len(cand) == 0:
            prev_nearest = -1
            continue
        d = np.array([geoms[c].distance(p) for c in cand])
        frac = np.array([geoms[c].project(p, normalized=True) for c in cand])
        order = np.argsort(d)
        rank = np.empty_like(order); rank[order] = np.arange(len(order))
        for j, c in enumerate(cand):
            connected = int(prev_nearest >= 0 and (c == prev_nearest or
                            len({u[c], v[c]} & {u[prev_nearest], v[prev_nearest]}) > 0))
            rows.append((tid, i, int(c), int(c == ref[i]),
                         d[j], int(rank[j]), d[j] - d.min(), len(cand),
                         angle_diff(gps_bear[i], arc_bear[c]), speed[i], gap[i],
                         arc_len[c], frac[j], connected))
        prev_nearest = int(cand[order[0]])

    cols = ["trip_id", "point_idx", "arc_id", "label",
            "dist", "dist_rank", "dist_minus_min", "n_cands",
            "heading_diff", "speed", "time_gap",
            "arc_len", "proj_frac", "conn_prev_nearest"]
    df = pd.DataFrame(rows, columns=cols)
    # points whose true arc is missing from the candidates (for candidate miss rate)
    df.attrs["n_points"] = n
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts, total_pts = [], 0
    for tid in sorted(os.listdir(a.data)):
        folder = os.path.join(a.data, tid)
        if not os.path.isdir(folder):
            continue
        df = build_trip(folder)
        total_pts += df.attrs["n_points"]
        parts.append(df)
        print(f"{tid}: {df.point_idx.nunique()} points with candidates, {len(df)} rows")
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_parquet(a.out, index=False)
    hit = out.groupby(["trip_id", "point_idx"]).label.max()
    print(f"\nrows={len(out)}  positive share={out.label.mean():.3f}")
    print(f"candidate miss rate = {1 - hit.sum() / total_pts:.3f}  (points with no correct candidate, incl. points with no candidates)")
