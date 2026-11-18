import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import CRS, Transformer
from shapely.geometry import LineString


def load_road_network(nodes_path, arcs_path):
    """
    Load road nodes and arcs, project coordinates into metres,
    and construct a LineString for every road arc.

    Node IDs and arc IDs are their original zero-based row indices.
    """

    nodes_raw = np.loadtxt(nodes_path, dtype=float)
    arcs_raw = np.loadtxt(arcs_path, dtype=int)

    nodes_raw = np.atleast_2d(nodes_raw)
    arcs_raw = np.atleast_2d(arcs_raw)

    if nodes_raw.shape[1] != 2:
        raise ValueError("Nodes file must contain longitude and latitude.")

    if arcs_raw.shape[1] != 2:
        raise ValueError("Arcs file must contain source and destination node IDs.")

    if not np.isfinite(nodes_raw).all():
        raise ValueError("Nodes contain invalid coordinates.")

    lon = nodes_raw[:, 0]
    lat = nodes_raw[:, 1]

    if np.any((lon < -180) | (lon > 180)) or np.any((lat < -90) | (lat > 90)):
        raise ValueError("Invalid longitude or latitude values.")

    node_ids = np.arange(len(nodes_raw))

    if np.any(arcs_raw < 0) or np.any(arcs_raw >= len(nodes_raw)):
        raise ValueError("Arc references a node ID outside the nodes array.")

    # Select the local UTM zone from the dataset's geographic centre.
    centre_lon = float(np.mean(lon))
    centre_lat = float(np.mean(lat))
    utm_zone = int((centre_lon + 180) // 6) + 1
    epsg = (32600 if centre_lat >= 0 else 32700) + utm_zone

    projected_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )

    x, y = transformer.transform(lon, lat)

    nodes = pd.DataFrame({
        "node_id": node_ids,
        "longitude": lon,
        "latitude": lat,
        "x": x,
        "y": y
    })

    arc_ids = np.arange(len(arcs_raw))
    geometries = []
    lengths = []

    for source, destination in arcs_raw:
        start = (x[source], y[source])
        end = (x[destination], y[destination])

        geometry = LineString([start, end])
        geometries.append(geometry)
        lengths.append(geometry.length)

    arcs = pd.DataFrame({
        "arc_id": arc_ids,
        "from_node_id": arcs_raw[:, 0],
        "to_node_id": arcs_raw[:, 1],
        "geometry": geometries,
        "length_m": lengths,
        "valid_length": np.asarray(lengths) > 0,
    })

    return {
        "nodes": nodes,
        "arcs": arcs,
        "crs": projected_crs.to_string()
    }


if __name__ == "__main__":
    trip_folder = Path("Student folder/train/00000001")

    network = load_road_network(
        trip_folder / "00000001.nodes",
        trip_folder / "00000001.arcs"
    )

    print("CRS:", network["crs"])
    print("Nodes:", len(network["nodes"]))
    print("Arcs:", len(network["arcs"]))
    print("First five arcs:")
    print(network["arcs"][["arc_id", "from_node_id", "to_node_id", "length_m", "valid_length"]].head())
    print("Valid length arcs:", int(network["arcs"]["valid_length"].sum()))
    print("Zero-length arcs (valid_length=False):", int((~network["arcs"]["valid_length"]).sum()))
    print("Invalid geometries:", sum(
        not geometry.is_valid for geometry in network["arcs"]["geometry"]
    ))
