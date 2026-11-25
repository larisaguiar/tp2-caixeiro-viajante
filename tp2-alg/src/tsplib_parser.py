import os
import numpy as np
from scipy.spatial import distance_matrix

class TSPLibParser:
    """
    Leitor de arquivos no formato TSPLIB para problemas do Caixeiro Viajante (TSP 2D).
    Suporta instâncias com coordenadas euclidianas 2D (EUC_2D).
    """

    @staticmethod
    def parse_file(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        metadata = {}
        coords = []
        in_coord_section = False

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if line_str == "NODE_COORD_SECTION":
                    in_coord_section = True
                    continue

                if line_str in ("EOF", "END"):
                    break

                if in_coord_section:
                    parts = line_str.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coords.append([x, y])
                        except ValueError:
                            continue
                else:
                    if ":" in line_str:
                        key, val = line_str.split(":", 1)
                        metadata[key.strip()] = val.strip()

        points = np.array(coords, dtype=np.float64)
        if len(points) == 0:
            raise ValueError(f"Nenhuma coordenada válida encontrada em {filepath}")

        dist_mat = distance_matrix(points, points)
        return points, dist_mat, metadata