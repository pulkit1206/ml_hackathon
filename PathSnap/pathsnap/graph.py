"""Directed road graph helpers: bounded shortest paths and route-continuity checks."""
import heapq

import numpy as np


class RoadGraph:
    def __init__(self, trip):
        self.u = trip["arcs"].u.to_numpy()
        self.v = trip["arcs"].v.to_numpy()
        self.length = trip["arc_len"]
        n_nodes = len(trip["nodes"])
        order = np.argsort(self.u, kind="stable")
        self.out_arcs = order                                   # arcs grouped by start node
        self.indptr = np.searchsorted(self.u[order], np.arange(n_nodes + 1))
        self._cache = {}

    def outgoing(self, node):
        return self.out_arcs[self.indptr[node]:self.indptr[node + 1]]

    def dists_from(self, src, cutoff):
        """Dijkstra from node `src`, stopping at `cutoff` metres. Returns {node: distance}."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf):
                continue
            for a in self.outgoing(n):
                nd = d + self.length[a]
                if nd <= cutoff:
                    m = self.v[a]
                    if nd < dist.get(m, np.inf):
                        dist[m] = nd
                        heapq.heappush(heap, (nd, m))
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[src] = (cutoff, dist)
        return dist

    def route_distance(self, a, fa, b, fb, cutoff, backtrack_tol=15.0):
        """Driving distance from fraction fa along arc a to fraction fb along arc b (inf if not reachable)."""
        if a == b:
            d = (fb - fa) * self.length[a]
            return max(d, 0.0) if d >= -backtrack_tol else np.inf
        rest = (1.0 - fa) * self.length[a]
        between = self.dists_from(self.v[a], cutoff).get(self.u[b], np.inf)
        return rest + between + fb * self.length[b]

    def continuous(self, a, b, max_hops=3):
        """True if b equals a or can be reached from a within `max_hops` arcs."""
        if a == b:
            return True
        frontier = {self.v[a]}
        for _ in range(max_hops):
            if self.u[b] in frontier:
                return True
            frontier = {self.v[x] for n in frontier for x in self.outgoing(n)}
        return False


def route_continuity(graph, arcs_seq):
    """Share of consecutive predicted-arc pairs that are connected (same arc or reachable within 3 arcs)."""
    arcs_seq = [a for a in arcs_seq if a >= 0]
    pairs = [(a, b) for a, b in zip(arcs_seq[:-1], arcs_seq[1:])]
    if not pairs:
        return float("nan")
    return float(np.mean([graph.continuous(a, b) for a, b in pairs]))
