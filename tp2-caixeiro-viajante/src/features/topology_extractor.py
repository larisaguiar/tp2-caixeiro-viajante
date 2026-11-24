import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors

class TopologyExtractor:
    """
    Extrator de características topológicas e geométricas de instâncias do TSP 2D.
    Extrai Aspect Ratio, Hull Ratio, 1-NN CV, Entropia Espacial e Métricas da AGM.
    """

    @staticmethod
    def extract_features(points: np.ndarray) -> dict:
        N = len(points)
        if N < 3:
            raise ValueError("Instância deve conter pelo menos 3 vértices.")

        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        dx = max_xy[0] - min_xy[0]
        dy = max_xy[1] - min_xy[1]
        aspect_ratio = float(max(dx, dy) / (min(dx, dy) + 1e-9))

        try:
            hull = ConvexHull(points)
            hull_ratio = float(len(hull.vertices) / N)
        except Exception:
            hull_ratio = 0.0

        nn = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nn.kneighbors(points)
        nn_distances = distances[:, 1]
        mean_nn = np.mean(nn_distances)
        nn_cv = float(np.std(nn_distances) / (mean_nn + 1e-9))

        grid_size = 5
        x_bins = np.linspace(min_xy[0], max_xy[0] + 1e-9, grid_size + 1)
        y_bins = np.linspace(min_xy[1], max_xy[1] + 1e-9, grid_size + 1)
        hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_bins, y_bins])
        probs = hist.flatten() / N
        probs = probs[probs > 0]
        spatial_entropy = float(-np.sum(probs * np.log2(probs)))

        dist_mat = distance_matrix(points, points)
        mst = minimum_spanning_tree(dist_mat).toarray()
        adjacency = (mst > 0) | (mst.T > 0)
        degrees = adjacency.sum(axis=0)
        max_degree = int(degrees.max())
        odd_vertices_count = int(np.sum(degrees % 2 != 0))
        odd_ratio = float(odd_vertices_count / N)

        return {
            'num_nodes': N,
            'aspect_ratio': aspect_ratio,
            'convex_hull_ratio': hull_ratio,
            'nn_cv': nn_cv,
            'spatial_entropy': spatial_entropy,
            'mst_max_degree': max_degree,
            'mst_odd_ratio': odd_ratio
        }