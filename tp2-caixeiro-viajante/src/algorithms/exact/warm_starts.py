import numpy as np
from scipy.spatial import ConvexHull
from src.algorithms.approx.twice_around_tree import twice_around_tree_tsp
from src.algorithms.approx.christofides import christofides_tsp

class WarmStartHeuristics:
    """
    Implementação das 5 estratégias de Warm-Start para o limite superior (Upper Bound) do Branch-and-Bound:
    1. Cold Start / Trivial (ordem sequencial de leitura)
    2. Vizinho Mais Próximo (Nearest Neighbor - Gulosa)
    3. Fecho Convexo (Convex Hull Insertion)
    4. Twice-around-the-tree (2-Aproximado)
    5. Christofides (1.5-Aproximado)
    """

    @staticmethod
    def cold_start(dist_matrix: np.ndarray):
        N = len(dist_matrix)
        route = list(range(N)) + [0]
        cost = sum(dist_matrix[route[i], route[i+1]] for i in range(N))
        return float(cost), route

    @staticmethod
    def nearest_neighbor(dist_matrix: np.ndarray):
        N = len(dist_matrix)
        unvisited = set(range(1, N))
        curr = 0
        route = [0]
        cost = 0.0

        while unvisited:
            next_node = min(unvisited, key=lambda x: dist_matrix[curr, x])
            cost += dist_matrix[curr, next_node]
            curr = next_node
            route.append(curr)
            unvisited.remove(curr)

        cost += dist_matrix[curr, 0]
        route.append(0)
        return float(cost), route

    @staticmethod
    def convex_hull_insertion(points: np.ndarray, dist_matrix: np.ndarray):
        N = len(points)
        if N < 3:
            return WarmStartHeuristics.nearest_neighbor(dist_matrix)

        try:
            hull = ConvexHull(points)
            subtour = list(hull.vertices)
        except Exception:
            return WarmStartHeuristics.nearest_neighbor(dist_matrix)

        uninserted = set(range(N)) - set(subtour)

        while uninserted:
            best_node = None
            best_insert_pos = None
            best_increase = float('inf')

            for k in uninserted:
                for i in range(len(subtour)):
                    u = subtour[i]
                    v = subtour[(i + 1) % len(subtour)]
                    increase = dist_matrix[u, k] + dist_matrix[k, v] - dist_matrix[u, v]
                    if increase < best_increase:
                        best_increase = increase
                        best_node = k
                        best_insert_pos = i + 1

            subtour.insert(best_insert_pos, best_node)
            uninserted.remove(best_node)

        subtour.append(subtour[0])
        cost = sum(dist_matrix[subtour[i], subtour[i+1]] for i in range(len(subtour)-1))
        return float(cost), subtour

    @staticmethod
    def twice_around_tree(dist_matrix: np.ndarray):
        return twice_around_tree_tsp(dist_matrix)

    @staticmethod
    def christofides(dist_matrix: np.ndarray):
        return christofides_tsp(dist_matrix)

    @classmethod
    def get_initial_ub(cls, strategy: str, points: np.ndarray, dist_matrix: np.ndarray):
        strat = strategy.lower()
        if strat == "cold_start":
            return cls.cold_start(dist_matrix)
        elif strat == "nearest_neighbor":
            return cls.nearest_neighbor(dist_matrix)
        elif strat == "convex_hull":
            return cls.convex_hull_insertion(points, dist_matrix)
        elif strat == "twice_around_tree":
            return cls.twice_around_tree(dist_matrix)
        elif strat == "christofides":
            return cls.christofides(dist_matrix)
        else:
            raise ValueError(f"Estratégia de warm-start inválida: {strategy}")