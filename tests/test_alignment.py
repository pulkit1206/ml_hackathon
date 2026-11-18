import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from src.candidates.alignment import (
    align_track_to_route,
    analyze_track_kinematics,
    validate_route_connectivity,
)
from src.candidates.generator import generate_candidates


def _utm_to_lonlat(x, y, crs="EPSG:32632"):
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def _make_alignment_network():
    """
    Create a directed network with:
    - arc 0: (470000, 5520000) -> (470000, 5520100)  (node 0 -> 1)
    - arc 1: (470000, 5520100) -> (470100, 5520100)  (node 1 -> 2)
    - arc 2: (470100, 5520100) -> (470100, 5520000)  (node 2 -> 3)
    - arc 3: (470100, 5520000) -> (470010, 5520000)  (node 3 -> 4, passes near node 0!)
    """
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4],
        "longitude": [0.0] * 5,
        "latitude": [0.0] * 5,
        "x": [470000.0, 470000.0, 470100.0, 470100.0, 470010.0],
        "y": [5520000.0, 5520100.0, 5520100.0, 5520000.0, 5520000.0],
    })
    arcs = pd.DataFrame({
        "arc_id": [0, 1, 2, 3],
        "from_node_id": [0, 1, 2, 3],
        "to_node_id": [1, 2, 3, 4],
        "geometry": [
            LineString([(470000.0, 5520000.0), (470000.0, 5520100.0)]),
            LineString([(470000.0, 5520100.0), (470100.0, 5520100.0)]),
            LineString([(470100.0, 5520100.0), (470100.0, 5520000.0)]),
            LineString([(470100.0, 5520000.0), (470010.0, 5520000.0)]),
        ],
        "length_m": [100.0, 100.0, 100.0, 90.0],
        "valid_length": [True, True, True, True],
    })
    return {"nodes": nodes, "arcs": arcs, "crs": "EPSG:32632"}


def test_route_connectivity_validation():
    network = _make_alignment_network()
    valid_res = validate_route_connectivity(network, np.array([0, 1, 2, 3]))
    assert valid_res["is_continuous_directed_path"] is True
    assert valid_res["directed_breaks"] == 0

    broken_res = validate_route_connectivity(network, np.array([0, 2, 3]))
    assert broken_res["is_continuous_directed_path"] is False
    assert broken_res["directed_breaks"] == 1


def test_monotonic_alignment_unmatched_and_ambiguous_cases():
    network = _make_alignment_network()
    route_arcs = np.array([0, 1, 2, 3], dtype="int64")

    # Obs 0: (470002, 5520050) -> clearly on arc 0 (2m away)
    # Obs 1: (470000, 5520100) -> exact corner between arc 0 and arc 1 (0m to both -> ambiguous!)
    # Obs 2: (470500, 5520500) -> far away (unmatched!)
    # Obs 3: (470098, 5520050) -> on arc 2 (2m away)
    # Obs 4: (470011, 5520002) -> 2m from arc 3, 11.2m from arc 0 (monotonic keeps arc 3, not arc 0)
    pts = [
        (470002.0, 5520050.0, 1.0),
        (470000.0, 5520100.0, 2.0),
        (470500.0, 5520500.0, 3.0),
        (470098.0, 5520050.0, 4.0),
        (470011.0, 5520002.0, 5.0),
    ]
    lon_lats = [_utm_to_lonlat(x, y) for x, y, _ in pts]
    track = pd.DataFrame({
        "point_id": [0, 1, 2, 3, 4],
        "longitude": [ll[0] for ll in lon_lats],
        "latitude": [ll[1] for ll in lon_lats],
        "timestamp": [p[2] for p in pts],
    })

    cands = generate_candidates(network, track, trip_id="00000001", radius_m=25)
    align_df, stats = align_track_to_route(
        network, track, cands, route_arcs, trip_id="00000001", ambiguity_tol_m=1.0
    )

    assert len(align_df) == 5
    assert align_df.loc[0, "alignment_status"] == "matched"
    assert align_df.loc[0, "aligned_arc_id"] == 0

    assert align_df.loc[1, "alignment_status"] == "ambiguous"
    assert pd.isna(align_df.loc[1, "aligned_arc_id"])

    assert align_df.loc[2, "alignment_status"] == "unmatched"
    assert pd.isna(align_df.loc[2, "aligned_arc_id"])

    assert align_df.loc[3, "alignment_status"] == "matched"
    assert align_df.loc[3, "aligned_arc_id"] == 2

    assert align_df.loc[4, "alignment_status"] == "matched"
    assert align_df.loc[4, "aligned_arc_id"] == 3

    assert stats["aligned_route_order_violations"] == 0


def test_kinematics_diagnostics_unrealistic_speed_and_jitter():
    network = _make_alignment_network()
    # Step 0->1: 0.3m move in 1s (stationary jitter)
    # Step 1->2: 100m move in 1s (unrealistic speed = 100 m/s > 50 m/s)
    # Step 2->3: 0m move in 1s (zero movement)
    pts = [
        (470000.0, 5520000.0, 1.0),
        (470000.3, 5520000.0, 2.0),
        (470100.3, 5520000.0, 3.0),
        (470100.3, 5520000.0, 4.0),
    ]
    lon_lats = [_utm_to_lonlat(x, y) for x, y, _ in pts]
    track = pd.DataFrame({
        "point_id": [0, 1, 2, 3],
        "longitude": [ll[0] for ll in lon_lats],
        "latitude": [ll[1] for ll in lon_lats],
        "timestamp": [p[2] for p in pts],
    })

    kin_stats = analyze_track_kinematics(network, track, unrealistic_speed_mps=50.0, jitter_dist_m=1.0)
    assert kin_stats["total_observations"] == 4
    assert kin_stats["dt_median_s"] == 1.0
    assert kin_stats["non_positive_dt_count"] == 0
    assert kin_stats["stationary_jitter_steps"] == 1
    assert kin_stats["unrealistic_speed_count"] == 1
    assert kin_stats["zero_movement_steps"] == 1
