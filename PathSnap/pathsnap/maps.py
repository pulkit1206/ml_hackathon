"""Folium maps: GPS observations, nearby road network, reference route, predicted routes."""
import folium
import numpy as np
import shapely


def _arc_lines(trip, arcs):
    u, v = trip["arcs"].u.to_numpy(), trip["arcs"].v.to_numpy()
    lat, lon = trip["nodes"].lat.to_numpy(), trip["nodes"].lon.to_numpy()
    return [[(lat[u[a]], lon[u[a]]), (lat[v[a]], lon[v[a]])] for a in arcs]


def trip_map(trip, pred_hmm=None, pred_clf=None, ref=None, matched=None, network_radius=60.0, title=""):
    tr = trip["track"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])

    # road network near the trajectory only (whole networks can have >100k arcs)
    pts = shapely.points(trip["px"], trip["py"])
    near = np.unique(trip["tree"].query(pts, predicate="dwithin", distance=network_radius)[1])
    net = folium.FeatureGroup("Road network (nearby)", show=True)
    folium.PolyLine(_arc_lines(trip, near), color="#9e9e9e", weight=2, opacity=0.6).add_to(net)
    net.add_to(m)

    if ref is not None:
        arcs = np.unique(ref[matched] if matched is not None else ref)
        g = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, arcs), color="#d62728", weight=7, opacity=0.5).add_to(g)
        g.add_to(m)
    if pred_clf is not None:
        g = folium.FeatureGroup("Predicted – classifier only", show=False)
        folium.PolyLine(_arc_lines(trip, np.unique(pred_clf[pred_clf >= 0])), color="#ff7f0e",
                        weight=4, opacity=0.9).add_to(g)
        g.add_to(m)
    if pred_hmm is not None:
        g = folium.FeatureGroup("Predicted – classifier + HMM", show=True)
        folium.PolyLine(_arc_lines(trip, np.unique(pred_hmm[pred_hmm >= 0])), color="#2ca02c",
                        weight=4, opacity=0.95).add_to(g)
        g.add_to(m)

    gps = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(tr) // 1500)              # keep the HTML light for long trips
    for la, lo in zip(tr.lat.to_numpy()[::step], tr.lon.to_numpy()[::step]):
        folium.CircleMarker((la, lo), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(m)
    folium.Marker((tr.lat.iloc[0], tr.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr.lat.iloc[-1], tr.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="black")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = f"""<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white;
      padding: 8px 12px; border: 1px solid #999; font-size: 13px;">
      <b>{title}</b><br>
      <span style="color:#1f77b4">&#9679;</span> GPS observations<br>
      <span style="color:#9e9e9e">&#9473;</span> Road network<br>
      {'<span style="color:#d62728">&#9473;</span> Reference route<br>' if ref is not None else ''}
      {'<span style="color:#ff7f0e">&#9473;</span> Classifier only<br>' if pred_clf is not None else ''}
      {'<span style="color:#2ca02c">&#9473;</span> Classifier + HMM' if pred_hmm is not None else ''}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
