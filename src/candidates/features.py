from pathlib import Path
import numpy as np
import pandas as pd

from src.candidates.generator import generate_candidates, load_track, project_track
from src.candidates.geometry import load_road_network

FEATURE_COLUMNS = [
    "trip_id",
    "point_id",
    "candidate_arc_id",
    "distance_m",
    "distance_rank",
    "arc_length_m",
    "along_arc_fraction",
    "candidate_count",
    "distance_gap_m",
    "gps_heading_deg",
    "road_heading_deg",
    "heading_diff_deg",
    "speed_mps",
    "connected_to_prev",
]


def _empty_features_df():
    return pd.DataFrame({
        "trip_id": pd.Series(dtype="object"),
        "point_id": pd.Series(dtype="int64"),
        "candidate_arc_id": pd.Series(dtype="int64"),
        "distance_m": pd.Series(dtype="float64"),
        "distance_rank": pd.Series(dtype="int64"),
        "arc_length_m": pd.Series(dtype="float64"),
        "along_arc_fraction": pd.Series(dtype="float64"),
        "candidate_count": pd.Series(dtype="int64"),
        "distance_gap_m": pd.Series(dtype="float64"),
        "gps_heading_deg": pd.Series(dtype="float64"),
        "road_heading_deg": pd.Series(dtype="float64"),
        "heading_diff_deg": pd.Series(dtype="float64"),
        "speed_mps": pd.Series(dtype="float64"),
        "connected_to_prev": pd.Series(dtype="bool"),
    })


def compute_heading_deg(dx, dy):
    """
    Compute compass heading in [0, 360) degrees (0=North, 90=East)
    from projected coordinate deltas (dx, dy). Returns NaN where movement is zero.
    """
    dx_arr = np.asarray(dx, dtype="float64")
    dy_arr = np.asarray(dy, dtype="float64")
    heading = np.full(dx_arr.shape, np.nan, dtype="float64")
    moved = (dx_arr != 0.0) | (dy_arr != 0.0)
    heading[moved] = np.degrees(np.arctan2(dx_arr[moved], dy_arr[moved])) % 360.0
    return heading


def normalize_heading_diff_deg(heading_a, heading_b):
    """
    Compute shortest angular difference between two headings in [0, 180] degrees.
    Preserves NaN where either heading is NaN.
    """
    a = np.asarray(heading_a, dtype="float64")
    b = np.asarray(heading_b, dtype="float64")
    diff = np.abs(a - b) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def compute_track_kinematics(projected_track):
    """
    Compute per-observation GPS heading (degrees) and speed (m/s)
    from consecutive observations within a single trip.

    Handles the first observation, duplicate/non-increasing timestamps,
    and zero movement safely (returns NaN for undefined heading/speed).
    """
    if projected_track.empty:
        return pd.DataFrame({
            "point_id": pd.Series(dtype="int64"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
            "gps_heading_deg": pd.Series(dtype="float64"),
            "speed_mps": pd.Series(dtype="float64"),
        })

    ordered = projected_track.sort_values("point_id").reset_index(drop=True)
    x = ordered["x"].to_numpy(dtype="float64")
    y = ordered["y"].to_numpy(dtype="float64")
    t = ordered["timestamp"].to_numpy(dtype="float64")

    dx = np.empty_like(x)
    dy = np.empty_like(y)
    dt = np.empty_like(t)

    dx[0] = 0.0
    dy[0] = 0.0
    dt[0] = 0.0
    if len(ordered) > 1:
        dx[1:] = x[1:] - x[:-1]
        dy[1:] = y[1:] - y[:-1]
        dt[1:] = t[1:] - t[:-1]

    gps_heading = compute_heading_deg(dx, dy)
    gps_heading[0] = np.nan

    step_dist = np.hypot(dx, dy)
    speed_mps = np.full(len(ordered), np.nan, dtype="float64")
    valid_dt = dt > 0.0
    valid_dt[0] = False
    speed_mps[valid_dt] = step_dist[valid_dt] / dt[valid_dt]

    return pd.DataFrame({
        "point_id": ordered["point_id"].to_numpy(dtype="int64"),
        "x": x,
        "y": y,
        "gps_heading_deg": gps_heading,
        "speed_mps": speed_mps,
    })


def compute_arc_attributes(network):
    """
    Extract arc_length_m, directed road_heading_deg, and endpoint coordinates
    from the projected road network, indexed by original arc_id.
    """
    arcs = network["arcs"]
    nodes = network["nodes"]

    node_x = nodes.set_index("node_id")["x"]
    node_y = nodes.set_index("node_id")["y"]

    x_from = arcs["from_node_id"].map(node_x).to_numpy(dtype="float64")
    y_from = arcs["from_node_id"].map(node_y).to_numpy(dtype="float64")
    x_to = arcs["to_node_id"].map(node_x).to_numpy(dtype="float64")
    y_to = arcs["to_node_id"].map(node_y).to_numpy(dtype="float64")

    dx = x_to - x_from
    dy = y_to - y_from
    road_heading = compute_heading_deg(dx, dy)

    return pd.DataFrame({
        "candidate_arc_id": arcs["arc_id"].to_numpy(dtype="int64"),
        "from_node_id": arcs["from_node_id"].to_numpy(dtype="int64"),
        "to_node_id": arcs["to_node_id"].to_numpy(dtype="int64"),
        "arc_length_m": arcs["length_m"].to_numpy(dtype="float64"),
        "x_from": x_from,
        "y_from": y_from,
        "dx_arc": dx,
        "dy_arc": dy,
        "road_heading_deg": road_heading,
    })


def compute_along_arc_fraction(px, py, x_from, y_from, dx_arc, dy_arc):
    """
    Vectorized normalized projection position in [0.0, 1.0] along a straight arc.
    Returns 0.0 for zero-length arcs.
    """
    denom = dx_arc * dx_arc + dy_arc * dy_arc
    frac = np.zeros_like(denom, dtype="float64")
    valid = denom > 0.0
    proj = ((px[valid] - x_from[valid]) * dx_arc[valid] + (py[valid] - y_from[valid]) * dy_arc[valid]) / denom[valid]
    frac[valid] = np.clip(proj, 0.0, 1.0)
    return frac


def compute_connectivity_to_prev(point_ids_ordered, cand_point_ids, cand_arc_ids, from_nodes, to_nodes):
    """
    Determine for each candidate row whether its arc is connected to at least one
    candidate arc from the immediately preceding observation (point_id - 1 in ordered track).

    An arc `a` at step t is connected if `a` was also a candidate at step t-1
    (staying on the same arc) or `from_node_id(a) == to_node_id(b)` for some
    candidate arc `b` at step t-1 (directed transition b -> a).
    """
    n_cands = len(cand_point_ids)
    connected = np.zeros(n_cands, dtype=bool)
    if n_cands == 0 or len(point_ids_ordered) == 0:
        return connected

    # Group candidate row indices by point_id for O(1) lookup per observation.
    obs_to_rows = {}
    for idx, pid in enumerate(cand_point_ids):
        obs_to_rows.setdefault(int(pid), []).append(idx)

    prev_arcs = set()
    prev_to_nodes = set()

    for step_idx, pid in enumerate(point_ids_ordered):
        row_indices = obs_to_rows.get(int(pid), [])
        if step_idx > 0 and row_indices and prev_arcs:
            for r_idx in row_indices:
                if int(cand_arc_ids[r_idx]) in prev_arcs or int(from_nodes[r_idx]) in prev_to_nodes:
                    connected[r_idx] = True

        # Update previous observation's candidate sets (empty if missing candidates).
        if row_indices:
            prev_arcs = {int(cand_arc_ids[r_idx]) for r_idx in row_indices}
            prev_to_nodes = {int(to_nodes[r_idx]) for r_idx in row_indices}
        else:
            prev_arcs = set()
            prev_to_nodes = set()

    return connected


def build_candidate_features(network, track_df, candidates_df):
    """
    Compute geometric, movement, and directed-connectivity features
    for a single trip's candidate DataFrame.
    """
    if candidates_df.empty:
        return _empty_features_df()

    projected_track = project_track(track_df, network["crs"])
    kinematics = compute_track_kinematics(projected_track)
    arc_attrs = compute_arc_attributes(network)

    df = candidates_df.copy()
    df["trip_id"] = df["trip_id"].astype(str)
    df["point_id"] = df["point_id"].astype("int64")
    df["candidate_arc_id"] = df["candidate_arc_id"].astype("int64")
    df["distance_m"] = df["distance_m"].astype("float64")
    df["distance_rank"] = df["distance_rank"].astype("int64")

    # Merge observation kinematics and arc attributes without altering row order.
    df = df.merge(kinematics, on="point_id", how="left", sort=False)
    df = df.merge(arc_attrs, on="candidate_arc_id", how="left", sort=False)

    # Geometric features.
    df["along_arc_fraction"] = compute_along_arc_fraction(
        df["x"].to_numpy(dtype="float64"),
        df["y"].to_numpy(dtype="float64"),
        df["x_from"].to_numpy(dtype="float64"),
        df["y_from"].to_numpy(dtype="float64"),
        df["dx_arc"].to_numpy(dtype="float64"),
        df["dy_arc"].to_numpy(dtype="float64"),
    )
    obs_group = df.groupby(["trip_id", "point_id"], sort=False)
    df["candidate_count"] = obs_group["candidate_arc_id"].transform("count").astype("int64")
    df["distance_gap_m"] = (df["distance_m"] - obs_group["distance_m"].transform("min")).astype("float64")

    # Movement features.
    df["heading_diff_deg"] = normalize_heading_diff_deg(
        df["gps_heading_deg"].to_numpy(dtype="float64"),
        df["road_heading_deg"].to_numpy(dtype="float64"),
    )

    # Directed connectivity to previous observation's candidates.
    df["connected_to_prev"] = compute_connectivity_to_prev(
        point_ids_ordered=kinematics["point_id"].to_numpy(dtype="int64"),
        cand_point_ids=df["point_id"].to_numpy(dtype="int64"),
        cand_arc_ids=df["candidate_arc_id"].to_numpy(dtype="int64"),
        from_nodes=df["from_node_id"].to_numpy(dtype="int64"),
        to_nodes=df["to_node_id"].to_numpy(dtype="int64"),
    )

    return df[FEATURE_COLUMNS].reset_index(drop=True)


def extract_trip_features(trip_folder, radius_m=50, output_path=None):
    """
    End-to-end feature extraction for a single trip at a chosen search radius.
    """
    trip_path = Path(trip_folder)
    trip_id = trip_path.name

    network = load_road_network(
        trip_path / f"{trip_id}.nodes",
        trip_path / f"{trip_id}.arcs",
    )
    track_df = load_track(trip_path / f"{trip_id}.track")
    candidates_df = generate_candidates(
        network=network,
        track_df=track_df,
        trip_id=trip_id,
        radius_m=radius_m,
    )
    features_df = build_candidate_features(network, track_df, candidates_df)

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_file, index=False)

    return features_df


if __name__ == "__main__":
    out_csv = Path("outputs/features/00000001_features_r50.csv")
    features = extract_trip_features(
        trip_folder="Student folder/train/00000001",
        radius_m=50,
        output_path=out_csv,
    )
    print("Feature rows:", len(features))
    print("Columns and dtypes:")
    print(features.dtypes)
    print("\nFirst 8 rows:")
    print(features.head(8).to_string(index=False))
    print("\nSummary statistics:")
    print(features.describe().to_string())
