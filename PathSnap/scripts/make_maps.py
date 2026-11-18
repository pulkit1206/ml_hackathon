"""Folium maps (GPS, nearby road network, reference route, classifier-only and HMM predictions).
Run after evaluate_hmm.py.

    python scripts/make_maps.py                 # all student-test trips
    python scripts/make_maps.py 00000052 00000007
"""
import sys

import numpy as np

from common import HMM_RES, MAPS, TEST_DIR
from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.maps import trip_map

ids = sys.argv[1:] or [d.name for d in list_trips(TEST_DIR)]
for tid in ids:
    f = HMM_RES / "preds" / f"{tid}.npz"
    if not f.exists():
        print(f"{tid}: no predictions (run evaluate_hmm.py first)"); continue
    p = np.load(f)
    trip = prepare_trip(load_trip(TEST_DIR / tid))
    m = p["matched"]
    acc_c = (p["pred_clf"][m] == p["ref"][m]).mean(); acc_h = (p["pred_hmm"][m] == p["ref"][m]).mean()
    title = f"Trip {tid} | accuracy: classifier {acc_c:.3f}, +HMM {acc_h:.3f}"
    trip_map(trip, p["pred_hmm"], p["pred_clf"], p["ref"], m, title=title).save(MAPS / f"student_test_{tid}.html")
    print(f"saved results/maps/student_test_{tid}.html  ({title})")
