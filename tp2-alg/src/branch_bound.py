
import time
import numpy as np

from src.warm_starts import WarmStartHeuristics


class BBNode:
    """Nó da árvore de busca do Branch-and-Bound."""

    def __init__(self, path, g_cost, lb, visited_mask):
        self.path = path
        self.g_cost = g_cost
        self.lb = lb
        self.visited_mask = visited_mask
        self.last_node = path[-1]


class BranchAndBoundTSP:
    """
    Branch-and-Bound para TSP simétrico.
    Estratégia de exploração: Depth-First Search (DFS).
    """

    def __init__(self, dist_matrix: np.ndarray, points: np.ndarray = None):

        self.dist_matrix = np.asarray(dist_matrix, dtype=float)
        self.points = points
        self.N = len(self.dist_matrix)

        if self.dist_matrix.ndim != 2:
            raise ValueError("A matriz de distâncias deve ser bidimensional.")

        if self.dist_matrix.shape[0] != self.dist_matrix.shape[1]:
            raise ValueError("A matriz de distâncias deve ser quadrada.")

        if not np.all(np.isfinite(self.dist_matrix)):
            raise ValueError("A matriz contém valores inválidos.")

        if np.any(self.dist_matrix < 0):
            raise ValueError("As distâncias devem ser não negativas.")

        if not np.allclose(self.dist_matrix, self.dist_matrix.T):
            raise ValueError("Esta implementação exige matriz simétrica.")

        # Pré-processamento: vizinhos ordenados pelo custo da aresta.
        self.sorted_neighbors = []

        for i in range(self.N):
            neighbors = [j for j in range(self.N) if j != i]

            neighbors.sort(
                key=lambda j: self.dist_matrix[i, j]
            )

            self.sorted_neighbors.append(neighbors)

    def _compute_lower_bound(self, path, g_cost, visited_mask):
        """
        Cota inferior baseada nas duas menores arestas disponíveis
        necessárias para completar o grau de cada vértice.
        """

        if len(path) == self.N:
            return (
                g_cost
                + self.dist_matrix[path[-1], path[0]]
            )

        def cheapest_edges(v, count, excluded=None):

            total = 0.0
            found = 0

            for u in self.sorted_neighbors[v]:

                if u == excluded:
                    continue

                total += self.dist_matrix[v, u]
                found += 1

                if found == count:
                    return total

            return float("inf")

        bound_sum = 2.0 * g_cost

        if len(path) == 1:
            # Raiz: ainda são necessárias duas arestas.
            bound_sum += cheapest_edges(path[0], 2)

        else:
            # Primeiro vértice: exclui a aresta já utilizada.
            bound_sum += cheapest_edges(
                path[0], 1, path[1]
            )

            # Último vértice: exclui a aresta já utilizada.
            bound_sum += cheapest_edges(
                path[-1], 1, path[-2]
            )

        # Cada vértice não visitado precisa de duas conexões.
        for v in range(self.N):

            if not (visited_mask & (1 << v)):
                bound_sum += cheapest_edges(v, 2)

        return bound_sum / 2.0

    def solve(
        self,
        warm_start_strategy="christofides",
        timeout_sec=300.0
    ):

        start_time = time.perf_counter()

        if self.N == 0:
            return 0.0, [], {
                "n_expanded": 0,
                "n_pruned": 0,
                "max_stack_size": 0,
                "max_heap_size": 0,
                "max_depth": 0,
                "timed_out": False,
                "stack_limit_reached": False,
                "completed": True,
                "optimality_proven": True,
                "warm_start_strategy": warm_start_strategy
            }

        if self.N == 1:
            return 0.0, [0, 0], {
                "n_expanded": 0,
                "n_pruned": 0,
                "max_stack_size": 0,
                "max_heap_size": 0,
                "max_depth": 1,
                "timed_out": False,
                "stack_limit_reached": False,
                "completed": True,
                "optimality_proven": True,
                "warm_start_strategy": warm_start_strategy
            }

        # 1. Warm-start: solução inicial e limite superior.
        ub_res = WarmStartHeuristics.get_initial_ub(
            warm_start_strategy,
            self.points,
            self.dist_matrix
        )

        best_cost = float(ub_res[0])
        best_route = ub_res[1]

        warm_sub_metrics = (
            ub_res[2] if len(ub_res) > 2 and isinstance(ub_res[2], dict)
            else {}
        )

        # 2. Inicialização da busca.
        initial_mask = 1 << 0

        initial_lb = self._compute_lower_bound(
            [0], 0.0, initial_mask
        )

        root_node = BBNode(
            path=[0],
            g_cost=0.0,
            lb=initial_lb,
            visited_mask=initial_mask
        )

        # DFS: pilha LIFO.
        stack = [root_node]

        # 3. Métricas.
        n_expanded = 0
        n_pruned = 0

        max_stack_size = 1
        max_depth = 1

        timed_out = False
        stack_limit_reached = False

        MAX_STACK_NODES = 1_000_000

        # 4. Busca em profundidade.
        while stack:

            elapsed = time.perf_counter() - start_time

            if elapsed >= timeout_sec:
                timed_out = True
                break

            if len(stack) >= MAX_STACK_NODES:
                stack_limit_reached = True
                break

            curr_node = stack.pop()
            n_expanded += 1

            # Poda por limite inferior.
            if curr_node.lb >= best_cost - 1e-9:
                n_pruned += 1
                continue

            max_depth = max(
                max_depth, len(curr_node.path)
            )

            # Solução completa.
            if len(curr_node.path) == self.N:

                final_cost = (
                    curr_node.g_cost
                    + self.dist_matrix[curr_node.last_node, 0]
                )

                if final_cost < best_cost:
                    best_cost = float(final_cost)
                    best_route = curr_node.path + [0]

                continue

            # 5. Ramificação.
            children = []

            for next_v in range(self.N):

                if curr_node.visited_mask & (1 << next_v):
                    continue

                new_g = (
                    curr_node.g_cost
                    + self.dist_matrix[curr_node.last_node, next_v]
                )

                new_mask = (
                    curr_node.visited_mask | (1 << next_v)
                )

                new_path = curr_node.path + [next_v]

                new_lb = self._compute_lower_bound(
                    new_path,
                    new_g,
                    new_mask
                )

                # Poda imediata.
                if new_lb >= best_cost - 1e-9:
                    n_pruned += 1
                    continue

                child = BBNode(
                    path=new_path,
                    g_cost=new_g,
                    lb=new_lb,
                    visited_mask=new_mask
                )

                children.append(child)

            # Coloca os piores filhos primeiro.
            # Como a pilha é LIFO, o melhor será explorado primeiro.
            children.sort(
                key=lambda node: (node.lb, node.g_cost),
                reverse=True
            )

            stack.extend(children)

            max_stack_size = max(
                max_stack_size, len(stack)
            )

        # 6. Estado final da execução.
        completed = not stack and not timed_out and not stack_limit_reached

        metrics = {
            "n_expanded": n_expanded,
            "n_pruned": n_pruned,
            "max_stack_size": max_stack_size,

            # Mantido para compatibilidade com análises antigas.
            "max_heap_size": max_stack_size,

            "max_depth": max_depth,
            "timed_out": timed_out,
            "stack_limit_reached": stack_limit_reached,
            "completed": completed,
            "optimality_proven": completed,
            "warm_start_strategy": warm_start_strategy,
            "elapsed_time": time.perf_counter() - start_time,
            **warm_sub_metrics
        }

        return best_cost, best_route, metrics
