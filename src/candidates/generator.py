from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.strtree import STRtree

from src.candidates.geometry import load_road_network

DEFAULT_RADII_M = (25, 50, 75, 100)

CANDIDATE_COLUMNS = [
    "trip_id",
    "point_id",
    "candidate_arc_id",
    "distance_m",
    "distance_rank",
]


def load_track(track_path):
    """
    Load GPS observations from a .track file.

    Expected columns: longitude, latitude, timestamp.
    Preserves the original zero-based observation index as point_id.
    """
    track_raw = np.loadtxt(track_path, dtype=float)
    if track_raw.size == 0:
        return pd.DataFrame({
            "point_id": pd.Series(dtype="int64"),
            "longitude": pd.Series(dtype="float64"),
            "latitude": pd.Series(dtype="float64"),
            "timestamp": pd.Series(dtype="float64"),
        })

    track_raw = np.atleast_2d(track_raw)
    if track_raw.shape[1] != 3:
        raise ValueError("Track file must contain longitude, latitude, and timestamp.")

    if not np.isfinite(track_raw).all():
        raise ValueError("Track contains invalid (NaN or infinite) values.")

    lon = track_raw[:, 0]
    lat = track_raw[:, 1]
    timestamp = track_raw[:, 2]

    if np.any((lon < -180) | (lon > 180)) or np.any((lat < -90) | (lat > 90)):
        raise ValueError("Track contains invalid longitude or latitude values.")

    return pd.DataFrame({
        "point_id": np.arange(len(track_raw), dtype="int64"),
        "longitude": lon,
        "latitude": lat,
        "timestamp": timestamp,
    })


def project_track(track_df, target_crs):
    """
    Project GPS track coordinates from EPSG:4326 into target_crs (metres).
    """
    if track_df.empty:
        projected = track_df.copy()
        projected["x"] = pd.Series(dtype="float64")
        projected["y"] = pd.Series(dtype="float64")
        return projected

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x, y = transformer.transform(
        track_df["longitude"].to_numpy(),
        track_df["latitude"].to_numpy(),
    )

    projected = track_df.copy()
    projected["x"] = x
    projected["y"] = y
    return projected


class ArcSpatialIndex:
    """
    Shapely STRtree spatial index over valid-length road arcs,
    preserving original zero-based arc_id values.
    """

    def __init__(self, arcs_df):
        if "valid_length" in arcs_df.columns:
            valid_arcs = arcs_df.loc[arcs_df["valid_length"]]
        else:
            valid_arcs = arcs_df.loc[arcs_df["length_m"] > 0]

        self.arc_ids = valid_arcs["arc_id"].to_numpy(dtype="int64")
        self.geometries = valid_arcs["geometry"].tolist()
        self._geom_id_to_pos = {id(geom): pos for pos, geom in enumerate(self.geometries)}
        self.tree = STRtree(self.geometries) if self.geometries else None

    def query_radius(self, point, radius_m):
        """
        Return list of (original_arc_id, distance_m) for all valid-length arcs
        within radius_m metres of point, sorted by (distance_m, arc_id).
        """
        if self.tree is None or radius_m < 0:
            return []

        try:
            hits = self.tree.query(point, predicate="dwithin", distance=radius_m)
        except TypeError:
            # Fallback for Shapely 1.x where predicate/distance kwargs are unsupported.
            hits = self.tree.query(point.buffer(radius_m))

        matches = []
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                pos = int(hit)
            else:
                pos = self._geom_id_to_pos[id(hit)]

            arc_geom = self.geometries[pos]
            dist_m = float(point.distance(arc_geom))
            if dist_m <= radius_m:
                matches.append((int(self.arc_ids[pos]), dist_m))

        matches.sort(key=lambda item: (item[1], item[0]))
        return matches


def generate_candidates(network, track_df, trip_id, radius_m, spatial_index=None):
    """
    Generate GPS-to-road candidate arcs within radius_m metres for each observation.

    Returns a DataFrame with columns:
        trip_id, point_id, candidate_arc_id, distance_m, distance_rank
    """
    trip_id_str = str(trip_id)
    if spatial_index is None:
        spatial_index = ArcSpatialIndex(network["arcs"])

    projected_track = project_track(track_df, network["crs"])
    if projected_track.empty or spatial_index.tree is None:
        return _empty_candidates_df()

    rows = []
    point_ids = projected_track["point_id"].to_numpy(dtype="int64")
    xs = projected_track["x"].to_numpy(dtype="float64")
    ys = projected_track["y"].to_numpy(dtype="float64")

    for point_id, x, y in zip(point_ids, xs, ys):
        pt = Point(float(x), float(y))
        matches = spatial_index.query_radius(pt, float(radius_m))
        for rank, (arc_id, dist_m) in enumerate(matches, start=1):
            rows.append({
                "trip_id": trip_id_str,
                "point_id": int(point_id),
                "candidate_arc_id": int(arc_id),
                "distance_m": float(dist_m),
                "distance_rank": int(rank),
            })

    if not rows:
        return _empty_candidates_df()

    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    df["trip_id"] = df["trip_id"].astype(str)
    df["point_id"] = df["point_id"].astype("int64")
    df["candidate_arc_id"] = df["candidate_arc_id"].astype("int64")
    df["distance_m"] = df["distance_m"].astype("float64")
    df["distance_rank"] = df["distance_rank"].astype("int64")
    return df


def _empty_candidates_df():
    return pd.DataFrame({
        "trip_id": pd.Series(dtype="object"),
        "point_id": pd.Series(dtype="int64"),
        "candidate_arc_id": pd.Series(dtype="int64"),
        "distance_m": pd.Series(dtype="float64"),
        "distance_rank": pd.Series(dtype="int64"),
    })


def summarize_candidates(candidates_df, total_observations, radius_m):
    """
    Compute candidate coverage and distance summary metrics for a single radius.
    """
    total_obs = int(total_observations)
    if total_obs == 0:
        return {
            "radius_m": radius_m,
            "total_observations": 0,
            "observations_with_candidates": 0,
            "observations_without_candidates": 0,
            "candidate_miss_rate": 0.0,
            "avg_candidates_per_observation": 0.0,
            "max_candidates_per_observation": 0,
            "min_candidate_distance_m": np.nan,
            "mean_candidate_distance_m": np.nan,
        }

    if candidates_df.empty:
        return {
            "radius_m": radius_m,
            "total_observations": total_obs,
            "observations_with_candidates": 0,
            "observations_without_candidates": total_obs,
            "candidate_miss_rate": 1.0,
            "avg_candidates_per_observation": 0.0,
            "max_candidates_per_observation": 0,
            "min_candidate_distance_m": np.nan,
            "mean_candidate_distance_m": np.nan,
        }

    counts_per_obs = candidates_df.groupby("point_id").size()
    obs_with_cands = int(len(counts_per_obs))
    obs_without_cands = total_obs - obs_with_cands

    return {
        "radius_m": radius_m,
        "total_observations": total_obs,
        "observations_with_candidates": obs_with_cands,
        "observations_without_candidates": obs_without_cands,
        "candidate_miss_rate": obs_without_cands / total_obs,
        "avg_candidates_per_observation": len(candidates_df) / total_obs,
        "max_candidates_per_observation": int(counts_per_obs.max()),
        "min_candidate_distance_m": float(candidates_df["distance_m"].min()),
        "mean_candidate_distance_m": float(candidates_df["distance_m"].mean()),
    }


def evaluate_trip_radii(trip_folder, radii=DEFAULT_RADII_M, output_dir=None):
    """
    Load a trip folder, generate candidates for each radius in radii,
    compute summary metrics, and optionally save CSV outputs.
    """
    trip_path = Path(trip_folder)
    trip_id = trip_path.name

    network = load_road_network(
        trip_path / f"{trip_id}.nodes",
        trip_path / f"{trip_id}.arcs",
    )
    track_df = load_track(trip_path / f"{trip_id}.track")
    spatial_index = ArcSpatialIndex(network["arcs"])

    all_candidates = []
    summary_rows = []

    for radius_m in radii:
        cands = generate_candidates(
            network=network,
            track_df=track_df,
            trip_id=trip_id,
            radius_m=radius_m,
            spatial_index=spatial_index,
        )
        summary_rows.append(
            summarize_candidates(cands, total_observations=len(track_df), radius_m=radius_m)
        )
        cands_with_radius = cands.copy()
        cands_with_radius["radius_m"] = radius_m
        all_candidates.append(cands_with_radius)

    candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame(columns=CANDIDATE_COLUMNS + ["radius_m"])
    )
    summary_df = pd.DataFrame(summary_rows)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        candidates_df.to_csv(out_path / f"{trip_id}_candidates.csv", index=False)
        summary_df.to_csv(out_path / f"{trip_id}_radius_summary.csv", index=False)

    return {
        "network": network,
        "track": track_df,
        "candidates": candidates_df,
        "summary": summary_df,
    }


if __name__ == "__main__":
    result = evaluate_trip_radii(
        trip_folder="Student folder/train/00000001",
        radii=DEFAULT_RADII_M,
        output_dir="outputs/candidates",
    )
    print("CRS:", result["network"]["crs"])
    print("Total GPS observations:", len(result["track"]))
    print("\nRadius comparison summary:")
    print(result["summary"].to_string(index=False))
    print("\nSample candidate records:")
    print(result["candidates"].head(10).to_string(index=False))
