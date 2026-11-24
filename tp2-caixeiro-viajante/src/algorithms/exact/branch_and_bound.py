import heapq
import time
import numpy as np
from src.algorithms.exact.warm_starts import WarmStartHeuristics

class BBNode:
    """
    Nó da árvore de busca do Branch-and-Bound para o TSP.
    """
    def __init__(self, path, g_cost, lb, visited_mask):
        self.path = path              # Lista de vértices visitados na ordem da sub-rota
        self.g_cost = g_cost          # Custo acumulado real da sub-rota
        self.lb = lb                  # Estimativa de Cota Inferior (Lower Bound)
        self.visited_mask = visited_mask # Bitmask de vértices visitados
        self.last_node = path[-1]     # Último vértice atingido

    def __lt__(self, other):
        # Fila de Prioridades (Min-Heap) prioriza menor Lower Bound (Best-First Search)
        if self.lb != other.lb:
            return self.lb < other.lb
        return len(self.path) > len(other.path) # Em desempate, prioriza nós mais profundos

class BranchAndBoundTSP:
    """
    Algoritmo Exato Branch-and-Bound para TSP com Busca Best-First (Min-Heap).
    """

    def __init__(self, dist_matrix: np.ndarray, points: np.ndarray = None):
        self.dist_matrix = dist_matrix
        self.points = points
        self.N = len(dist_matrix)

        # Pré-calcular as duas arestas de menor custo para cada vértice
        self.min_edges = []
        for i in range(self.N):
            sorted_row = np.sort(dist_matrix[i])
            sorted_row = sorted_row[sorted_row > 1e-9]
            if len(sorted_row) >= 2:
                self.min_edges.append((sorted_row[0], sorted_row[1]))
            elif len(sorted_row) == 1:
                self.min_edges.append((sorted_row[0], sorted_row[0]))
            else:
                self.min_edges.append((0.0, 0.0))

    def _compute_lower_bound(self, path, g_cost, visited_mask):
        """
        Calcula a cota inferior baseada no custo acumulado + metade das duas arestas mínimas dos vértices não fixados.
        """
        if len(path) == self.N:
            return g_cost + self.dist_matrix[path[-1], path[0]]

        bound_sum = 2.0 * g_cost

        # Aresta de entrada do nó raiz e aresta de saída do nó atual
        bound_sum += self.min_edges[path[0]][0]
        bound_sum += self.min_edges[path[-1]][0]

        # Para vértices intermediários não visitados
        for v in range(self.N):
            if not (visited_mask & (1 << v)):
                bound_sum += self.min_edges[v][0] + self.min_edges[v][1]

        return bound_sum / 2.0

    def solve(self, warm_start_strategy: str = "christofides", timeout_sec: float = 300.0):
        """
        Executa a busca Branch-and-Bound com inicialização de limite superior (Warm-Start).
        """
        start_time = time.perf_counter()

        # 1. Warm-Start: Obter limite superior inicial (Upper Bound)
        # DEPOIS:
        ub_res = WarmStartHeuristics.get_initial_ub(
            warm_start_strategy, self.points, self.dist_matrix
        )
        best_cost, best_route = ub_res[0], ub_res[1]

        # 2. Inicializar Nó Raiz (Vértice 0)
        initial_mask = 1 << 0
        initial_lb = self._compute_lower_bound([0], 0.0, initial_mask)
        root_node = BBNode(path=[0], g_cost=0.0, lb=initial_lb, visited_mask=initial_mask)

        # Fila de Prioridades (Min-Heap)
        heap = [root_node]

        # Métricas da Árvore de Busca
        n_expanded = 0
        n_pruned = 0
        max_heap_size = 1
        max_depth = 1
        timed_out = False

        # Limite máximo de nós na Heap para evitar OOM (Out of Memory)
        MAX_HEAP_NODES = 1_000_000

        while heap:
            # Checar Timeout de Tempo ou Estouro de Memória (Heap Size)
            if (time.perf_counter() - start_time > timeout_sec) or (len(heap) > MAX_HEAP_NODES):
                timed_out = True
                break

            curr_node = heapq.heappop(heap)
            n_expanded += 1

            # Poda por Limite Superior Atual
            if curr_node.lb >= best_cost - 1e-9:
                n_pruned += 1
                continue

            max_depth = max(max_depth, len(curr_node.path))

            # Caso Nó Folha: Solução Completa
            if len(curr_node.path) == self.N:
                final_cost = curr_node.g_cost + self.dist_matrix[curr_node.last_node, 0]
                if final_cost < best_cost:
                    best_cost = float(final_cost)
                    best_route = curr_node.path + [0]
                continue

            # Ramificação (Branching)
            for next_v in range(self.N):
                if not (curr_node.visited_mask & (1 << next_v)):
                    new_g = curr_node.g_cost + self.dist_matrix[curr_node.last_node, next_v]
                    new_mask = curr_node.visited_mask | (1 << next_v)
                    new_path = curr_node.path + [next_v]

                    new_lb = self._compute_lower_bound(new_path, new_g, new_mask)

                    # Poda Imediata ao Criar o Filho
                    if new_lb < best_cost - 1e-9:
                        child = BBNode(path=new_path, g_cost=new_g, lb=new_lb, visited_mask=new_mask)
                        heapq.heappush(heap, child)
                    else:
                        n_pruned += 1

            max_heap_size = max(max_heap_size, len(heap))

        metrics = {
            'n_expanded': n_expanded,
            'n_pruned': n_pruned,
            'max_heap_size': max_heap_size,
            'max_depth': max_depth,
            'timed_out': timed_out,
            'warm_start_strategy': warm_start_strategy
        }

        return best_cost, best_route, metrics