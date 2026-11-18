"""Classifier-only vs classifier + HMM/Viterbi on the 10 student-test trips, using the exported model
and decoder parameters (exactly what the API runs).

    python scripts/evaluate_hmm.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ART, HMM_RES, TEST_DIR
from pathsnap.io import list_trips
from pathsnap.pipeline import Matcher, score_trip

matcher = Matcher(ART)
(HMM_RES / "preds").mkdir(exist_ok=True)
rows = []
for tdir in list_trips(TEST_DIR):
    r = matcher.predict_trip(tdir)
    s, ref, matched = score_trip(tdir, r)
    np.savez(HMM_RES / "preds" / f"{tdir.name}.npz", pred_hmm=r["pred_hmm"], pred_clf=r["pred_clf"],
             ref=ref, matched=matched)
    rows.append({"trip_id": tdir.name, **s, "chain_breaks": r["hmm_stats"]["breaks"], "runtime_s": r["runtime_s"]})
    print(f"{tdir.name}: clf {s['accuracy_clf']:.4f} -> hmm {s['accuracy_hmm']:.4f} | continuity "
          f"{s['continuity_clf']:.3f} -> {s['continuity_hmm']:.3f} | breaks {r['hmm_stats']['breaks']} | {r['runtime_s']}s")

df = pd.DataFrame(rows)
w = df.n_scored_points
overall = {"trip_id": "ALL (point-weighted)", "n_points": df.n_points.sum(), "n_scored_points": w.sum(),
           **{c: np.average(df[c], weights=w) for c in ["accuracy_clf", "accuracy_hmm", "continuity_clf",
                                                         "continuity_hmm", "route_arc_precision", "route_arc_recall"]},
           "chain_breaks": df.chain_breaks.sum(), "runtime_s": df.runtime_s.sum()}
df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
df.to_csv(HMM_RES / "student_test_clf_vs_hmm.csv", index=False)
print("\n" + df.round(4).to_string(index=False))

d = df.iloc[:-1]
x = np.arange(len(d)); bw = 0.4
for metric, fname, title in (("accuracy", "accuracy", "Point-level accuracy"), ("continuity", "continuity", "Route continuity")):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - bw / 2, d[f"{metric}_clf"], bw, label="classifier only", color="#ff7f0e")
    ax.bar(x + bw / 2, d[f"{metric}_hmm"], bw, label="classifier + HMM", color="#2ca02c")
    ax.set_xticks(x, d.trip_id, rotation=30); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
    ax.set_title(f"Student-test {title.lower()} per trip"); plt.tight_layout()
    plt.savefig(HMM_RES / f"clf_vs_hmm_{fname}.png", dpi=150); plt.close()
