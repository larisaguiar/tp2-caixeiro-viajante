import numpy as np
import time
import tracemalloc
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def twice_around_tree_tsp(dist_matrix: np.ndarray):
    """
    Algoritmo 2-Aproximado para TSP (Twice-around-the-tree)
    com Profiling em Nível de Subrotina para a Árvore Geradora Mínima (AGM).

    Returns:
        tuple: (cost, route, subroutine_metrics)
    """
    N = len(dist_matrix)
    subroutine_metrics = {
        'mst_time_sec': 0.0,
        'mst_peak_mem_mb': 0.0,
        'dfs_time_sec': 0.0
    }

    if N == 0:
        return 0.0, [], subroutine_metrics
    if N == 1:
        return 0.0, [0, 0], subroutine_metrics

    # --- SUBROTINA 1: Árvore Geradora Mínima (AGM) ---
    tracemalloc.start()
    t0 = time.perf_counter()
    graph = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(graph).toarray()
    t1 = time.perf_counter()
    _, mst_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    subroutine_metrics['mst_time_sec'] = float(t1 - t0)
    subroutine_metrics['mst_peak_mem_mb'] = float(mst_bytes / (1024.0 * 1024.0))

    adj = [[] for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if mst[u, v] > 0 or mst[v, u] > 0:
                adj[u].append(v)

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
    subroutine_metrics['dfs_time_sec'] = float(t3 - t2)

    cost = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route) - 1))
    return float(cost), route, subroutine_metrics