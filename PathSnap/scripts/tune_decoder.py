"""Tune the HMM decoder (beta, w_emit) on TRAINING trips without leakage.

For each tuning trip, classifier probabilities come from the GroupKFold model that did NOT see that trip
(out-of-fold), so the decoder is tuned on realistic, unseen-trip probabilities. Student-test is not used here.

    python scripts/tune_decoder.py            # 20 random training trips
    python scripts/tune_decoder.py 40
"""
import itertools
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from common import ART, HMM_RES, N_SPLITS, SEED, TEAM_LABELS, TRAIN_DIR, TRAIN_PQ, selected_features
from pathsnap.features import candidate_table, prepare_trip
from pathsnap.graph import RoadGraph
from pathsnap.hmm import HMMParams, build_lattice, classifier_only, decode
from pathsnap.io import load_trip
from pathsnap.labels import TeamLabels, reference_labels

N_TRIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BETAS = [2, 5, 10, 20, 50]
W_EMIT = [0.25, 0.5, 1.0, 2.0]
base = HMMParams()

feats = selected_features()
best = json.load(open(ART / "best_params.json"))
train = pd.read_parquet(TRAIN_PQ)
fold_of, models = {}, []
for k, (tr, va) in enumerate(GroupKFold(n_splits=N_SPLITS).split(train, groups=train.trip_id)):
    for t in train.iloc[va].trip_id.unique():
        fold_of[t] = k
    models.append(LGBMClassifier(**best, random_state=SEED, verbose=-1).fit(train.iloc[tr][feats], train.iloc[tr].label))
    print(f"fold model {k} fitted")

rng = np.random.default_rng(SEED)
trips = sorted(fold_of)
chosen = sorted(rng.choice(trips, size=min(N_TRIPS, len(trips)), replace=False))
team = TeamLabels(TEAM_LABELS)
rows = []
for tid in chosen:
    t0 = time.perf_counter()
    trip = prepare_trip(load_trip(TRAIN_DIR / tid))
    ref, matched, _ = reference_labels(trip, team)
    cand = candidate_table(trip)
    p = models[fold_of[tid]].predict_proba(cand[feats])[:, 1]
    g = RoadGraph(trip)
    lat = build_lattice(trip, cand, p, base, g)
    clf = classifier_only(trip, cand, p)
    n_m = int(matched.sum())
    rows.append({"trip_id": tid, "beta": None, "w_emit": None, "n_scored": n_m,
                 "correct": int((clf[matched] == ref[matched]).sum())})
    for b, w in itertools.product(BETAS, W_EMIT):
        pred, _ = decode(lat, b, w)
        rows.append({"trip_id": tid, "beta": b, "w_emit": w, "n_scored": n_m,
                     "correct": int((pred[matched] == ref[matched]).sum())})
    print(f"{tid}: {len(ref)} pts, lattice+grid in {time.perf_counter()-t0:.1f}s")

df = pd.DataFrame(rows)
df.to_csv(HMM_RES / "decoder_tuning_per_trip.csv", index=False)
clf_acc = df[df.beta.isna()].correct.sum() / df[df.beta.isna()].n_scored.sum()
grid = df.dropna(subset=["beta"]).groupby(["beta", "w_emit"])[["correct", "n_scored"]].sum()
grid["accuracy"] = grid.correct / grid.n_scored
grid = grid.reset_index()
grid.to_csv(HMM_RES / "decoder_tuning_grid.csv", index=False)
top = grid.sort_values("accuracy", ascending=False).iloc[0]
params = HMMParams(beta=float(top.beta), w_emit=float(top.w_emit))
json.dump(params.to_dict(), open(ART / "hmm_params.json", "w"), indent=2)
print(grid.pivot(index="beta", columns="w_emit", values="accuracy").round(4))
print(f"\nclassifier-only accuracy on tuning trips: {clf_acc:.4f}")
print(f"best decoder: beta={top.beta}, w_emit={top.w_emit} -> accuracy {top.accuracy:.4f}  (saved artifacts/hmm_params.json)")

piv = grid.pivot(index="beta", columns="w_emit", values="accuracy")
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
ax.set_xlabel("w_emit (weight of classifier evidence)"); ax.set_ylabel("beta (m)")
ax.set_title(f"HMM accuracy on {len(chosen)} training trips (out-of-fold)\nclassifier only = {clf_acc:.3f}")
plt.colorbar(im); plt.tight_layout(); plt.savefig(HMM_RES / "decoder_tuning_heatmap.png", dpi=150)
