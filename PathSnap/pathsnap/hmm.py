"""HMM + Viterbi route reconstruction.

Hidden state at GPS point t = one of its candidate arcs (the arc id is kept with every state,
so the state -> road-segment correspondence is preserved).

Emission   log B_a(o_t)  = w_emit * log p_clf(a | o_t)          (p_clf = trained classifier probability)
Transition log A(a -> b) = -| d_route(a -> b) - d_gps(t-1, t) | / beta
           Newson & Krumm (2009): the driving distance between the two snapped positions should be close
           to the straight-line distance between the two GPS fixes. d_route = inf (impossible) if b cannot
           be reached from a by driving along directed arcs within the search cutoff.
Viterbi    delta_t(b) = max_a [ delta_{t-1}(a) + log A(a -> b) ] + log B_b(o_t)

Edge cases
  * point with no candidates  -> no prediction (-1); the chain links the neighbouring points that do have candidates
  * all transitions impossible -> the chain is broken and restarted at t (delta_t = emissions only)
  * GPS gap longer than max_gap_s -> chain broken and restarted (the route through a long gap is not guessed)
  * the driving-distance search cutoff grows with the GPS distance, so moderate gaps are still bridged

The lattice (candidates + raw transition costs |d_route - d_gps|) is built once per trip; decoding for
different (beta, w_emit) values is then cheap, which makes decoder tuning fast.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .graph import RoadGraph


@dataclass
class HMMParams:
    beta: float = 10.0        # metres; tolerance between driving and straight-line distance
    w_emit: float = 1.0       # weight of classifier evidence relative to transitions
    top_k: int = 10           # candidates per point kept as states (highest classifier probability)
    max_gap_s: float = 60.0   # break the chain after a longer GPS gap
    eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


def build_lattice(trip, cand, proba, params=HMMParams(), graph=None):
    graph = graph or RoadGraph(trip)
    c = cand[["point_idx", "arc_id", "proj_frac"]].copy()
    c["p"] = proba
    c = c.sort_values(["point_idx", "p"], ascending=[True, False]).groupby("point_idx").head(params.top_k)
    steps = [(int(pi), g.arc_id.to_numpy(), g.proj_frac.to_numpy(), g.p.to_numpy())
             for pi, g in c.groupby("point_idx", sort=True)]
    t, px, py = trip["track"].t.to_numpy(float), trip["px"], trip["py"]
    costs = [None]
    for s in range(1, len(steps)):
        pa, a_prev, f_prev, _ = steps[s - 1]
        pb, a_cur, f_cur, _ = steps[s]
        cost = np.full((len(a_prev), len(a_cur)), np.inf)
        if t[pb] - t[pa] <= params.max_gap_s:
            d_gps = float(np.hypot(px[pb] - px[pa], py[pb] - py[pa]))
            cutoff = min(2.0 * d_gps + 150.0, 3000.0)
            for i, (a, fa) in enumerate(zip(a_prev, f_prev)):
                for j, (b, fb) in enumerate(zip(a_cur, f_cur)):
                    d = graph.route_distance(a, fa, b, fb, cutoff)
                    if np.isfinite(d):
                        cost[i, j] = abs(d - d_gps)
        costs.append(cost)
    return {"n": len(px), "steps": steps, "costs": costs}


def decode(lattice, beta=10.0, w_emit=1.0, eps=1e-6):
    steps, costs = lattice["steps"], lattice["costs"]
    pred = np.full(lattice["n"], -1, dtype=np.int64)
    if not steps:
        return pred, {"breaks": 0, "steps": 0}
    emits = [w_emit * np.log(np.maximum(s[3], eps)) for s in steps]
    segments, idxs, backs = [], [0], [None]
    delta, breaks = emits[0], 0
    for s in range(1, len(steps)):
        scores = delta[:, None] - costs[s] / beta
        bp = scores.argmax(0)
        best = scores[bp, np.arange(scores.shape[1])]
        if not np.isfinite(best).any():
            breaks += 1
            segments.append((idxs, backs, delta))
            idxs, backs, delta = [s], [None], emits[s]
        else:
            idxs.append(s); backs.append(bp); delta = best + emits[s]
    segments.append((idxs, backs, delta))
    for idxs, backs, last in segments:
        k = int(np.argmax(last))
        for pos in range(len(idxs) - 1, -1, -1):
            s = idxs[pos]
            pred[steps[s][0]] = steps[s][1][k]
            if backs[pos] is not None:
                k = int(backs[pos][k])
    return pred, {"breaks": breaks, "steps": len(steps)}


def viterbi_decode(trip, cand, proba, params=HMMParams(), graph=None):
    lat = build_lattice(trip, cand, proba, params, graph)
    return decode(lat, params.beta, params.w_emit, params.eps)


def classifier_only(trip, cand, proba):
    """Top-1 classifier pick for every point (baseline without sequence information)."""
    pred = np.full(len(trip["px"]), -1, dtype=np.int64)
    if len(cand) == 0:
        return pred
    tmp = cand[["point_idx", "arc_id"]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby("point_idx").p.idxmax()]
    pred[best.point_idx.to_numpy()] = best.arc_id.to_numpy()
    return pred
