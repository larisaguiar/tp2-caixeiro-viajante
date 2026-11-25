import numpy as np
import time
import tracemalloc
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _run_subroutine(func, *args, **kwargs):
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()

    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()

    _, peak_bytes = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    return res, float(t1 - t0), float(peak_bytes / (1024.0 * 1024.0))


def twice_around_tree_tsp(dist_matrix: np.ndarray):
    """
    Algoritmo 2-Aproximado para TSP (Twice-around-the-tree)
    com Profiling em Nível de Subrotina para a Árvore Geradora Mínima (AGM).

    Returns:
        tuple: (cost, route, subroutine_metrics)
    """
    N = len(dist_matrix)
    subroutine_metrics = {
        "mst_time_sec": 0.0,
        "mst_peak_mem_mb": 0.0,
        "dfs_time_sec": 0.0,
    }

    if N == 0:
        return 0.0, [], subroutine_metrics
    if N == 1:
        return 0.0, [0, 0], subroutine_metrics

    # --- SUBROTINA 1: Árvore Geradora Mínima (AGM) ---
    def compute_mst():
        graph = csr_matrix(dist_matrix)
        return minimum_spanning_tree(graph)

    mst_sparse, mst_time, mst_mem = _run_subroutine(compute_mst)
    subroutine_metrics["mst_time_sec"] = mst_time
    subroutine_metrics["mst_peak_mem_mb"] = mst_mem

    adj = [[] for _ in range(N)]
    rows, cols = mst_sparse.nonzero()
    for u, v in zip(rows, cols):
        adj[u].append(int(v))
        adj[v].append(int(u))

    # --- SUBROTINA 2: DFS Pré-Ordem + Shortcutting ---
    t2 = time.perf_counter()
    visited = [False] * N
    route = []

    def dfs(u):
        visited[u] = True
        route.append(u)
        for neighbor in adj[u]:
            if not visited[neighbor]:
                dfs(neighbor)

    dfs(0)
    route.append(route[0])
    t3 = time.perf_counter()
    subroutine_metrics["dfs_time_sec"] = float(t3 - t2)

    cost = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    return float(cost), route, subroutine_metrics