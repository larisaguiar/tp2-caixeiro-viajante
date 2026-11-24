import numpy as np
import networkx as nx
import time
import tracemalloc
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def christofides_tsp(dist_matrix: np.ndarray):
    """
    Algoritmo 1.5-Aproximado de Christofides para TSP Métrico/Euclidiano
    com Profiling em Nível de Subrotina (AGM vs. Matching de Edmonds vs. Circuito Euleriano).

    Returns:
        tuple: (cost, route, subroutine_metrics)
    """
    N = len(dist_matrix)
    subroutine_metrics = {
        'mst_time_sec': 0.0,
        'mst_peak_mem_mb': 0.0,
        'matching_time_sec': 0.0,
        'matching_peak_mem_mb': 0.0,
        'eulerian_time_sec': 0.0,
        'num_odd_nodes': 0
    }

    if N == 0:
        return 0.0, [], subroutine_metrics
    if N <= 2:
        route = list(range(N)) + [0]
        cost = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        return float(cost), route, subroutine_metrics

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

    adj_mst = [[] for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if mst[u, v] > 0 or mst[v, u] > 0:
                adj_mst[u].append(v)

    # --- SUBROTINA 2: Identificação de Vértices Ímpares ---
    degrees = [len(adj_mst[u]) for u in range(N)]
    odd_nodes = [u for u in range(N) if degrees[u] % 2 != 0]
    subroutine_metrics['num_odd_nodes'] = len(odd_nodes)

    # --- SUBROTINA 3: Emparelhamento Perfeito de Peso Mínimo (Edmonds Blossom via NetworkX) ---
    tracemalloc.start()
    t2 = time.perf_counter()
    G_odd = nx.Graph()
    for i in range(len(odd_nodes)):
        for j in range(i + 1, len(odd_nodes)):
            u = odd_nodes[i]
            v = odd_nodes[j]
            G_odd.add_edge(u, v, weight=dist_matrix[u, v])

    matching = nx.min_weight_matching(G_odd, weight='weight')
    t3 = time.perf_counter()
    _, matching_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    subroutine_metrics['matching_time_sec'] = float(t3 - t2)
    subroutine_metrics['matching_peak_mem_mb'] = float(matching_bytes / (1024.0 * 1024.0))

    # --- SUBROTINA 4: Circuito Euleriano e Shortcutting ---
    t4 = time.perf_counter()
    multi_g = nx.MultiGraph()
    for u in range(N):
        for v in adj_mst[u]:
            if u < v:
                multi_g.add_edge(u, v, weight=dist_matrix[u, v])
    for u, v in matching:
        multi_g.add_edge(u, v, weight=dist_matrix[u, v])

    euler_circuit = list(nx.eulerian_circuit(multi_g, source=0))
    visited = set()
    route = []

    for u, v in euler_circuit:
        if u not in visited:
            visited.add(u)
            route.append(u)

    route.append(route[0])
    t5 = time.perf_counter()
    subroutine_metrics['eulerian_time_sec'] = float(t5 - t4)

    cost = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route) - 1))
    return float(cost), route, subroutine_metrics