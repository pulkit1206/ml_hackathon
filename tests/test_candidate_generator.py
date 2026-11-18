import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from src.candidates.generator import (
    CANDIDATE_COLUMNS,
    DEFAULT_RADII_M,
    ArcSpatialIndex,
    generate_candidates,
    project_track,
    summarize_candidates,
)


def _make_synthetic_network():
    """
    Create a synthetic road network in EPSG:32632 with:
    - arc_id 0: zero-length arc (valid_length=False) at (470000, 5520000)
    - arc_id 1: horizontal line y=5520000, x in [470000, 470100]
    - arc_id 2: zero-length arc (valid_length=False) at (470050, 5520030)
    - arc_id 3: horizontal line y=5520030, x in [470000, 470100]
    - arc_id 4: horizontal line y=5520080, x in [470000, 470100]
    """
    arcs = pd.DataFrame({
        "arc_id": [0, 1, 2, 3, 4],
        "from_node_id": [0, 0, 2, 2, 4],
        "to_node_id": [0, 1, 2, 3, 5],
        "geometry": [
            LineString([(470000.0, 5520000.0), (470000.0, 5520000.0)]),
            LineString([(470000.0, 5520000.0), (470100.0, 5520000.0)]),
            LineString([(470050.0, 5520030.0), (470050.0, 5520030.0)]),
            LineString([(470000.0, 5520030.0), (470100.0, 5520030.0)]),
            LineString([(470000.0, 5520080.0), (470100.0, 5520080.0)]),
        ],
        "length_m": [0.0, 100.0, 0.0, 100.0, 100.0],
        "valid_length": [False, True, False, True, True],
    })
    return {"nodes": pd.DataFrame(), "arcs": arcs, "crs": "EPSG:32632"}


def _utm_to_lonlat(x, y, crs="EPSG:32632"):
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def test_zero_length_arc_exclusion_and_arc_id_preservation():
    network = _make_synthetic_network()
    # Point directly on arc 0 (zero-length) and arc 1 (y=5520000), 10m north -> y=5520010.
    lon, lat = _utm_to_lonlat(470050.0, 5520010.0)
    track = pd.DataFrame({
        "point_id": [0],
        "longitude": [lon],
        "latitude": [lat],
        "timestamp": [1.0],
    })

    cands = generate_candidates(network, track, trip_id="00000001", radius_m=25)
    assert list(cands.columns) == CANDIDATE_COLUMNS
    # Arc 0 and Arc 2 are zero-length and must be excluded.
    assert 0 not in cands["candidate_arc_id"].values
    assert 2 not in cands["candidate_arc_id"].values
    # Arc 1 is 10m away, Arc 3 is 20m away. Original arc_ids (1 and 3) must be preserved.
    assert list(cands["candidate_arc_id"]) == [1, 3]
    assert list(cands["distance_rank"]) == [1, 2]
    assert np.allclose(cands["distance_m"], [10.0, 20.0], atol=1e-4)
    assert cands["trip_id"].iloc[0] == "00000001"
    assert cands["point_id"].iloc[0] == 0


def test_crs_consistency():
    network = _make_synthetic_network()
    expected_x, expected_y = 470025.0, 5520015.0
    lon, lat = _utm_to_lonlat(expected_x, expected_y, crs=network["crs"])
    track = pd.DataFrame({
        "point_id": [0],
        "longitude": [lon],
        "latitude": [lat],
        "timestamp": [1.0],
    })
    projected = project_track(track, network["crs"])
    assert np.isclose(projected["x"].iloc[0], expected_x, atol=1e-4)
    assert np.isclose(projected["y"].iloc[0], expected_y, atol=1e-4)


def test_empty_candidate_results():
    network = _make_synthetic_network()
    # Far away from all arcs (500m south).
    lon, lat = _utm_to_lonlat(470050.0, 5519500.0)
    track = pd.DataFrame({
        "point_id": [0],
        "longitude": [lon],
        "latitude": [lat],
        "timestamp": [1.0],
    })

    cands = generate_candidates(network, track, trip_id="00000001", radius_m=25)
    assert cands.empty
    assert list(cands.columns) == CANDIDATE_COLUMNS

    summary = summarize_candidates(cands, total_observations=len(track), radius_m=25)
    assert summary["total_observations"] == 1
    assert summary["observations_with_candidates"] == 0
    assert summary["observations_without_candidates"] == 1
    assert summary["candidate_miss_rate"] == 1.0
    assert summary["avg_candidates_per_observation"] == 0.0
    assert summary["max_candidates_per_observation"] == 0
    assert np.isnan(summary["min_candidate_distance_m"])
    assert np.isnan(summary["mean_candidate_distance_m"])


def test_multiple_candidates_and_all_four_radii():
    network = _make_synthetic_network()
    # Point at y=5520005:
    # - dist to arc 1 (y=5520000) = 5m
    # - dist to arc 3 (y=5520030) = 25m
    # - dist to arc 4 (y=5520080) = 75m
    # Plus a second point far away (0 candidates).
    lon1, lat1 = _utm_to_lonlat(470050.0, 5520005.0)
    lon2, lat2 = _utm_to_lonlat(470050.0, 5519000.0)
    track = pd.DataFrame({
        "point_id": [0, 1],
        "longitude": [lon1, lon2],
        "latitude": [lat1, lat2],
        "timestamp": [1.0, 2.0],
    })

    index = ArcSpatialIndex(network["arcs"])
    counts_by_radius = {}
    for r in DEFAULT_RADII_M:
        cands = generate_candidates(network, track, trip_id="00000001", radius_m=r, spatial_index=index)
        counts_by_radius[r] = len(cands)
        if not cands.empty:
            assert (cands["distance_m"] <= r + 1e-9).all()
            assert list(cands["distance_rank"]) == list(range(1, len(cands) + 1))

    assert DEFAULT_RADII_M == (25, 50, 75, 100)
    # At r=25: arc 1 (5m) and arc 3 (25m) -> 2 candidates
    assert counts_by_radius[25] >= 1
    # At r=75 and r=100: arc 1, arc 3, arc 4 -> 3 candidates
    assert counts_by_radius[100] == 3
