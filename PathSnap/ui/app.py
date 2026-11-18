"""PathSnap evaluator UI (runs on Laptop 2).

    PATHSNAP_API=http://<laptop-1-ip>:8000 streamlit run ui/app.py

The UI only sends the selected trip id to the API and displays what comes back.
It has no access to the model, the trip files or the reference routes.
"""
import os
import time

import folium
import requests
import streamlit as st

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap – map-matching evaluation")

api = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000"))
try:
    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
    st.sidebar.success(f"Connected – {len(trips)} trips available")
except Exception as e:
    st.sidebar.error(f"Cannot reach the API at {api}: {e}")
    st.stop()

trip_id = st.radio("Select an evaluation trip", trips, horizontal=True)
if st.button("Run map-matching", type="primary"):
    with st.spinner(f"Sending trip {trip_id} to the API…"):
        t0 = time.perf_counter()
        r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=600)
        elapsed = time.perf_counter() - t0
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    res = r.json()
    st.caption(f"Response from {api}/predict in {elapsed:.1f}s (server processing {res['runtime_s']}s)")

    m = res["metrics"]
    if m:
        c = st.columns(5)
        c[0].metric("Accuracy (classifier + HMM)", f"{m['accuracy_hmm']*100:.2f}%")
        c[1].metric("Accuracy (classifier only)", f"{m['accuracy_clf']*100:.2f}%")
        c[2].metric("Route continuity (HMM)", f"{m['continuity_hmm']*100:.1f}%")
        c[3].metric("GPS points scored", f"{m['n_scored_points']} / {m['n_points']}")
        c[4].metric("Route arc recall", f"{m['route_arc_recall']*100:.1f}%")
    else:
        st.warning("No reference route on the server for this trip – metrics unavailable.")

    gps = res["gps"]
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=15, tiles="OpenStreetMap")
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    g1 = folium.FeatureGroup("Predicted route (classifier + HMM)")
    folium.PolyLine(res["predicted_route_segments"], color="#2ca02c", weight=5).add_to(g1)
    g1.add_to(fmap)
    g2 = folium.FeatureGroup("GPS observations")
    for p in gps:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, weight=0, fill_opacity=0.8).add_to(g2)
    g2.add_to(fmap)
    start, end = res.get("start", gps[0]), res.get("end", gps[-1])
    folium.Marker(start, tooltip="Start", popup="Start of trip",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fmap)
    folium.Marker(end, tooltip="End", popup="End of trip",
                  icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    html = fmap.get_root().render()
    if hasattr(st, "iframe"):                       # newer Streamlit
        st.iframe(html, height=620)
    else:                                           # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=620)
    st.caption("Blue = raw GPS observations, green line = predicted route returned by the API, green pin = start, red pin = end.")
