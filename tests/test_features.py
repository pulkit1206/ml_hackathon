import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from src.candidates.features import (
    FEATURE_COLUMNS,
    build_candidate_features,
    compute_heading_deg,
    normalize_heading_diff_deg,
)
from src.candidates.generator import generate_candidates


def _utm_to_lonlat(x, y, crs="EPSG:32632"):
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def _make_synthetic_network():
    """
    Create a directed road network in EPSG:32632:
    - node 0: (470000, 5520000)
    - node 1: (470000, 5520100)  [100m North of node 0]
    - node 2: (470100, 5520100)  [100m East of node 1]
    - node 3: (470100, 5520000)  [100m South of node 2]
    - node 4: (470500, 5520500)
    - node 5: (470500, 5520600)

    Arcs:
    - arc 10: 0 -> 1 (North, heading 0 deg, length 100m)
    - arc 11: 1 -> 2 (East, heading 90 deg, length 100m)
    - arc 12: 2 -> 1 (West, heading 270 deg, length 100m; reverse of arc 11)
    - arc 13: 4 -> 5 (North, disconnected from 0->1->2)
    - arc 14: 0 -> 0 (zero-length arc, valid_length=False)
    """
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5],
        "longitude": [0.0] * 6,
        "latitude": [0.0] * 6,
        "x": [470000.0, 470000.0, 470100.0, 470100.0, 470500.0, 470500.0],
        "y": [5520000.0, 5520100.0, 5520100.0, 5520000.0, 5520500.0, 5520600.0],
    })
    arcs = pd.DataFrame({
        "arc_id": [10, 11, 12, 13, 14],
        "from_node_id": [0, 1, 2, 4, 0],
        "to_node_id": [1, 2, 1, 5, 0],
        "geometry": [
            LineString([(470000.0, 5520000.0), (470000.0, 5520100.0)]),
            LineString([(470000.0, 5520100.0), (470100.0, 5520100.0)]),
            LineString([(470100.0, 5520100.0), (470000.0, 5520100.0)]),
            LineString([(470500.0, 5520500.0), (470500.0, 5520600.0)]),
            LineString([(470000.0, 5520000.0), (470000.0, 5520000.0)]),
        ],
        "length_m": [100.0, 100.0, 100.0, 100.0, 0.0],
        "valid_length": [True, True, True, True, False],
    })
    return {"nodes": nodes, "arcs": arcs, "crs": "EPSG:32632"}


def test_heading_and_wraparound_normalization():
    # North, East, South, West, zero-movement
    h = compute_heading_deg([0.0, 10.0, 0.0, -10.0, 0.0], [10.0, 0.0, -10.0, 0.0, 0.0])
    assert np.allclose(h[:4], [0.0, 90.0, 180.0, 270.0])
    assert np.isnan(h[4])

    # Wrap-around difference: 350 vs 10 -> 20 degrees, 0 vs 180 -> 180 degrees
    diffs = normalize_heading_diff_deg([350.0, 10.0, 0.0, np.nan], [10.0, 350.0, 180.0, 90.0])
    assert np.allclose(diffs[:3], [20.0, 20.0, 180.0])
    assert np.isnan(diffs[3])


def test_geometric_movement_and_connectivity_features():
    network = _make_synthetic_network()
    # Obs 0: (470005, 5520025) at t=0s -> near arc 10 (5m east, 25% along arc)
    # Obs 1: (470005, 5520075) at t=5s -> moved 50m North (heading 0 deg, speed 10 m/s), near arc 10 & 11/12
    # Obs 2: (470050, 5520105) at t=10s -> moved ~(45, 30) NE, near arc 11 (1->2) & arc 12 (2->1)
    pts = [
        (470005.0, 5520025.0, 0.0),
        (470005.0, 5520075.0, 5.0),
        (470050.0, 5520105.0, 10.0),
    ]
    lon_lats = [_utm_to_lonlat(x, y) for x, y, _ in pts]
    track = pd.DataFrame({
        "point_id": [0, 1, 2],
        "longitude": [ll[0] for ll in lon_lats],
        "latitude": [ll[1] for ll in lon_lats],
        "timestamp": [p[2] for p in pts],
    })

    cands = generate_candidates(network, track, trip_id="00000001", radius_m=30)
    feats = build_candidate_features(network, track, cands)

    assert list(feats.columns) == FEATURE_COLUMNS
    assert (feats["trip_id"] == "00000001").all()

    # Observation 0: first observation -> gps_heading_deg & speed_mps are NaN, connected_to_prev is False
    p0 = feats[feats["point_id"] == 0]
    assert len(p0) == 1
    assert p0["candidate_arc_id"].iloc[0] == 10
    assert np.isclose(p0["along_arc_fraction"].iloc[0], 0.25, atol=1e-4)
    assert p0["candidate_count"].iloc[0] == 1
    assert np.isclose(p0["distance_gap_m"].iloc[0], 0.0)
    assert np.isnan(p0["gps_heading_deg"].iloc[0])
    assert np.isnan(p0["speed_mps"].iloc[0])
    assert np.isnan(p0["heading_diff_deg"].iloc[0])
    assert bool(p0["connected_to_prev"].iloc[0]) is False

    # Observation 1: moved 50m North in 5s -> heading ~0 deg, speed ~10 m/s, connected to arc 10
    p1 = feats[feats["point_id"] == 1]
    arc10_p1 = p1[p1["candidate_arc_id"] == 10].iloc[0]
    assert np.isclose(arc10_p1["gps_heading_deg"], 0.0, atol=1e-3)
    assert np.isclose(arc10_p1["speed_mps"], 10.0, atol=1e-3)
    assert np.isclose(arc10_p1["heading_diff_deg"], 0.0, atol=1e-3)
    assert bool(arc10_p1["connected_to_prev"]) is True

    # Observation 2: arc 11 (1->2) and arc 12 (2->1) are both 5m away.
    # Previous obs (point_id=1) had arc 10 (0->1) and arc 11 (1->2) and arc 12 (2->1).
    # Note: arc 11 is connected from arc 10 (to_node=1 == from_node=1) and also present in p1.
    p2 = feats[feats["point_id"] == 2]
    assert set(p2["candidate_arc_id"]) == {11, 12}
    assert (p2["candidate_count"] == 2).all()
    assert np.allclose(p2["distance_gap_m"], [0.0, 0.0], atol=1e-4)


def test_duplicate_timestamps_zero_movement_and_missing_prev_candidates():
    network = _make_synthetic_network()
    # Obs 0: near arc 10
    # Obs 1: far away (no candidates) -> breaks connectivity for Obs 2
    # Obs 2: near arc 11 at t=10
    # Obs 3: identical coords & timestamp as Obs 2 -> zero movement & duplicate timestamp
    pts = [
        (470005.0, 5520025.0, 0.0),
        (471000.0, 5521000.0, 5.0),
        (470050.0, 5520105.0, 10.0),
        (470050.0, 5520105.0, 10.0),
    ]
    lon_lats = [_utm_to_lonlat(x, y) for x, y, _ in pts]
    track = pd.DataFrame({
        "point_id": [0, 1, 2, 3],
        "longitude": [ll[0] for ll in lon_lats],
        "latitude": [ll[1] for ll in lon_lats],
        "timestamp": [p[2] for p in pts],
    })

    cands = generate_candidates(network, track, trip_id="00000001", radius_m=20)
    feats = build_candidate_features(network, track, cands)

    # Obs 1 had no candidates, so Obs 2 must have connected_to_prev == False.
    p2 = feats[feats["point_id"] == 2]
    assert not p2["connected_to_prev"].any()

    # Obs 3 has duplicate timestamp (dt=0) and zero movement (dx=0, dy=0)
    p3 = feats[feats["point_id"] == 3]
    assert p3["speed_mps"].isna().all()
    assert p3["gps_heading_deg"].isna().all()
    assert p3["heading_diff_deg"].isna().all()
    # Still connected because Obs 2 had the same candidate arcs.
    assert p3["connected_to_prev"].all()


def test_zero_length_arc_in_features_and_empty_candidates():
    network = _make_synthetic_network()
    lon, lat = _utm_to_lonlat(470000.0, 5520000.0)
    track = pd.DataFrame({
        "point_id": [0],
        "longitude": [lon],
        "latitude": [lat],
        "timestamp": [1.0],
    })

    # Empty candidates DataFrame returns empty features DataFrame with exact schema.
    empty_cands = generate_candidates(network, track, trip_id="00000001", radius_m=-1)
    empty_feats = build_candidate_features(network, track, empty_cands)
    assert empty_feats.empty
    assert list(empty_feats.columns) == FEATURE_COLUMNS

    # Direct check if a zero-length arc (arc_id=14) were passed in candidates.
    manual_cand = pd.DataFrame({
        "trip_id": ["00000001"],
        "point_id": [0],
        "candidate_arc_id": [14],
        "distance_m": [0.0],
        "distance_rank": [1],
    })
    z_feats = build_candidate_features(network, track, manual_cand)
    assert z_feats["arc_length_m"].iloc[0] == 0.0
    assert z_feats["along_arc_fraction"].iloc[0] == 0.0
    assert np.isnan(z_feats["road_heading_deg"].iloc[0])
