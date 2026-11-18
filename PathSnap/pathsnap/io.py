"""Readers for the raw trip files (.track, .nodes, .arcs, .route). All files are headerless."""
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path, names):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, found {df.shape[1]}")
    df.columns = names
    return df


def load_trip(folder, with_route=True):
    """Return a dict with track / nodes / arcs (and route if present and requested)."""
    folder = Path(folder)
    tid = folder.name
    trip = {
        "trip_id": tid,
        "track": _read(folder / f"{tid}.track", ["lon", "lat", "t"]),
        "nodes": _read(folder / f"{tid}.nodes", ["lon", "lat"]),
        "arcs": _read(folder / f"{tid}.arcs", ["u", "v"]).astype(np.int64),
    }
    route_path = folder / f"{tid}.route"
    trip["route"] = (pd.read_csv(route_path, header=None)[0].to_numpy(np.int64)
                     if with_route and route_path.exists() else None)
    return trip


def list_trips(partition_dir):
    p = Path(partition_dir)
    return sorted(d for d in p.iterdir() if d.is_dir()) if p.is_dir() else []
