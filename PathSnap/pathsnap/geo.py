"""Per-trip metric projection.

Trips are spread over the whole world (about -100 to +137 degrees longitude, both hemispheres),
so a single fixed projection (e.g. one UTM zone) would distort distances and directions.
Each trip gets its own azimuthal-equidistant projection centred on the trip's GPS points,
the same approach as the team's label-generation notebook.
"""
import numpy as np
from pyproj import CRS, Transformer


def local_transformer(lon, lat):
    lon0, lat0 = float(np.nanmean(lon)), float(np.nanmean(lat))
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0:.10f} +lon_0={lon0:.10f} +datum=WGS84 +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project_trip(trip):
    """Add metric x/y coordinates (metres) to track and nodes."""
    tr = local_transformer(trip["track"].lon.to_numpy(), trip["track"].lat.to_numpy())
    px, py = tr.transform(trip["track"].lon.to_numpy(), trip["track"].lat.to_numpy())
    nx, ny = tr.transform(trip["nodes"].lon.to_numpy(), trip["nodes"].lat.to_numpy())
    trip["px"], trip["py"] = np.asarray(px), np.asarray(py)
    trip["nx"], trip["ny"] = np.asarray(nx), np.asarray(ny)
    return trip


def bearing(dx, dy):
    """Direction in degrees, 0 = north (+y), clockwise, 0..360."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
