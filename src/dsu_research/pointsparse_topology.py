"""A1-DESIGN: data-independent, degree-controlled PointSparse graphs."""

import hashlib

import numpy as np
import torch

from .compartment_ablation import base_topology

D, H, K = 784, 128, 64
MIX_ATTEMPTS = 100_000
OPT_ATTEMPTS = 60_000


def graph_hash(graph):
    """SHA-256 of contiguous little-endian int64 indices in row order."""
    values = np.asarray(graph, dtype="<i8")
    if values.shape != (H, K):
        raise ValueError("expected [128,64] graph")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _degrees(graph):
    values = np.asarray(graph)
    if values.shape != (H, K) or np.any((values < 0) | (values >= D)):
        raise ValueError("invalid graph shape or indices")
    if any(len(set(row)) != K for row in values):
        raise ValueError("duplicate edge in hidden unit")
    return np.bincount(values.ravel(), minlength=D)


def _balanced_base(seed):
    rng = np.random.default_rng(seed)
    # Consecutive blocks of 64 have no duplicate. Every input appears 10 or 11 times.
    labels = rng.permutation(D)
    graph = labels[np.arange(H * K).reshape(H, K) % D].copy()
    adjacency = np.zeros((H, D), dtype=bool)
    adjacency[np.arange(H)[:, None], graph] = True
    accepted = 0
    for _ in range(MIX_ATTEMPTS):
        u, v = rng.choice(H, 2, replace=False)
        a, b = rng.integers(K, size=2)
        x, y = graph[u, a], graph[v, b]
        if adjacency[u, y] or adjacency[v, x]:
            continue
        graph[u, a], graph[v, b] = y, x
        adjacency[u, x] = adjacency[v, y] = False
        adjacency[u, y] = adjacency[v, x] = True
        accepted += 1
    return graph, accepted


def _objective(overlap):
    upper = overlap[np.triu_indices(H, 1)].astype(np.float64)
    return float(np.mean(upper / (2 * K - upper)))


def _optimize(base, seed, direction):
    rng = np.random.default_rng(seed ^ 0xA1A1A1A1 ^ (0 if direction < 0 else 0x3A3A3A3A))
    graph = base.copy()
    adjacency = np.zeros((H, D), dtype=bool)
    adjacency[np.arange(H)[:, None], graph] = True
    overlap = adjacency.astype(np.int16) @ adjacency.astype(np.int16).T
    initial = _objective(overlap)
    accepted = 0
    for _ in range(OPT_ATTEMPTS):
        u, v = rng.choice(H, 2, replace=False)
        a, b = rng.integers(K, size=2)
        x, y = graph[u, a], graph[v, b]
        if adjacency[u, y] or adjacency[v, x]:
            continue
        du = adjacency[:, y].astype(np.int16) - adjacency[:, x].astype(np.int16)
        dv = -du
        du[[u, v]] = 0
        dv[[u, v]] = 0
        old_u = overlap[u]
        old_v = overlap[v]
        delta = (np.sum((old_u + du) / (2 * K - old_u - du) - old_u / (2 * K - old_u)) +
                 np.sum((old_v + dv) / (2 * K - old_v - dv) - old_v / (2 * K - old_v)))
        if direction * delta <= 1e-12:
            continue
        graph[u, a], graph[v, b] = y, x
        adjacency[u, x] = adjacency[v, y] = False
        adjacency[u, y] = adjacency[v, x] = True
        overlap[u] += du
        overlap[:, u] += du
        overlap[v] += dv
        overlap[:, v] += dv
        accepted += 1
    return graph, accepted, initial, _objective(overlap)


def make_topology(arm, topology_seed):
    """Return G64 and auditable generation metadata; never reads data or weights."""
    if type(topology_seed) is not int or not 0 <= topology_seed <= 2**32 - 1:
        raise ValueError("topology_seed must be a nonnegative 32-bit integer")
    if arm not in ("A0", "A1", "A2", "A3"):
        raise ValueError("unknown A1 topology arm")
    if arm == "A0":
        graph = base_topology(topology_seed).numpy()
        return torch.from_numpy(graph.copy()), {"base_hash": graph_hash(graph),
            "final_hash": graph_hash(graph), "mix_swaps": 0, "objective_swaps": 0,
            "objective_initial": structural_metrics(graph)["mean_jaccard"],
            "objective_final": structural_metrics(graph)["mean_jaccard"]}
    base, mixed = _balanced_base(topology_seed)
    base_hash = graph_hash(base)
    if arm == "A1":
        graph, swaps = base, 0
        initial = final = structural_metrics(base)["mean_jaccard"]
    else:
        graph, swaps, initial, final = _optimize(base, topology_seed, -1 if arm == "A2" else 1)
    degrees = _degrees(graph)
    if not (np.count_nonzero(degrees == 11) == 352 and np.count_nonzero(degrees == 10) == 432):
        raise AssertionError("degree invariant broken")
    return torch.from_numpy(graph.astype(np.int64)), {"base_hash": base_hash,
        "final_hash": graph_hash(graph), "mix_swaps": mixed, "objective_swaps": swaps,
        "objective_initial": initial, "objective_final": final}


def structural_metrics(graph):
    """Describe coverage, degrees, pair overlap, and bipartite connectivity."""
    values = np.asarray(graph)
    degrees = _degrees(values)
    adjacency = np.zeros((H, D), dtype=np.int16)
    adjacency[np.arange(H)[:, None], values] = 1
    counts = (adjacency @ adjacency.T)[np.triu_indices(H, 1)]
    jaccard = counts / (2 * K - counts)
    # All balanced graphs have positive input degrees. Connectivity is computed
    # on the bipartite graph, including all input and hidden vertices.
    seen_h, seen_x = {0}, set()
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for x in values[u]:
            x = int(x)
            if x in seen_x:
                continue
            seen_x.add(x)
            for v in np.flatnonzero(adjacency[:, x]):
                v = int(v)
                if v not in seen_h:
                    seen_h.add(v)
                    frontier.append(v)
    return {"coverage": int(np.count_nonzero(degrees)), "input_degree_mean": float(degrees.mean()),
        "input_degree_sd": float(degrees.std()), "input_degree_min": int(degrees.min()),
        "input_degree_max": int(degrees.max()), "input_degree_counts":
        {str(k): int(v) for k, v in zip(*np.unique(degrees, return_counts=True))},
        "hidden_degree": K, "mean_jaccard": float(jaccard.mean()),
        "median_jaccard": float(np.median(jaccard)), "max_jaccard": float(jaccard.max()),
        "jaccard_sd": float(jaccard.std()), "jaccard_p90": float(np.percentile(jaccard, 90)),
        "overlap_count_distribution": {str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
        "connected": len(seen_h) == H and len(seen_x) == D}
