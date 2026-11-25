import argparse
import sys

from src.tsplib_parser import TSPLibParser
from src.topology_extractor import TopologyExtractor
from src.twice_around_tree import twice_around_tree_tsp
from src.christofides import christofides_tsp
from src.branch_bound import BranchAndBoundTSP
from src.metrics import MetricsTracker


def main():
    parser = argparse.ArgumentParser(description="TP2 - Caixeiro Viajante (TSP)")
    parser.add_argument("--instance", type=str, required=True, help="Caminho para o arquivo .tsp")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["twice_around_tree", "christofides", "branch_and_bound"],
        required=True,
        help="Algoritmo a executar",
    )
    parser.add_argument(
        "--warm_start",
        type=str,
        default="christofides",
        choices=["cold_start", "nearest_neighbor", "convex_hull", "twice_around_tree", "christofides"],
        help="Estratégia de Warm-Start para o Branch-and-Bound",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout para o Branch-and-Bound (segundos)")

    args = parser.parse_args()

    # 1. Carregar Instância
    try:
        points, dist_mat, meta = TSPLibParser.parse_file(args.instance)
    except Exception as e:
        print(f"Erro ao carregar arquivo de instância: {e}")
        sys.exit(1)

    print("=" * 50)
    print("  TP2 - Execução de Algoritmo para o TSP")
    print("=" * 50)
    print(f"Instância: {args.instance}")
    print(f"Algoritmo: {args.algo}")
    if args.algo == "branch_and_bound":
        print(f"Warm-Start: {args.warm_start}")

    # 2. Extrair Características Topológicas
    feats = TopologyExtractor.extract_features(points)
    print("-" * 50)
    print("Características Topológicas:")
    print(f"  - Nós (N): {feats['num_nodes']}")
    print(f"  - Aspect Ratio: {feats['aspect_ratio']:.4f}")
    print(f"  - Convex Hull Ratio: {feats['convex_hull_ratio']:.4f}")
    print(f"  - 1-NN CV: {feats['nn_cv']:.4f}")
    print(f"  - Entropia Espacial: {feats['spatial_entropy']:.4f}")
    print("-" * 50)

    # 3. Execução Profiling
    if args.algo == "twice_around_tree":
        (cost, route, sub_metrics), w_time, p_mem = MetricsTracker.run_with_profiling(
            twice_around_tree_tsp, dist_mat
        )
        print("RESULTADOS DA EXECUÇÃO:")
        print(f"  - Custo da Rota Encontrada: {cost:.4f}")
        print(f"  - Tempo Total (Wall Clock): {w_time:.4f} seg")
        print(f"  - Pico de Memória (RAM): {p_mem:.4f} MB")
        print(f"  - Tempo AGM: {sub_metrics['mst_time_sec']:.6f} seg | Memória AGM: {sub_metrics['mst_peak_mem_mb']:.4f} MB")
        print(f"  - Tempo DFS: {sub_metrics['dfs_time_sec']:.6f} seg")
        print(f"  - Sequência da Rota: {route[:10]}... (Total: {len(route)} vértices)")

    elif args.algo == "christofides":
        (cost, route, sub_metrics), w_time, p_mem = MetricsTracker.run_with_profiling(
            christofides_tsp, dist_mat
        )
        print("RESULTADOS DA EXECUÇÃO:")
        print(f"  - Custo da Rota Encontrada: {cost:.4f}")
        print(f"  - Tempo Total (Wall Clock): {w_time:.4f} seg")
        print(f"  - Pico de Memória (RAM): {p_mem:.4f} MB")
        print(f"  - Tempo AGM: {sub_metrics['mst_time_sec']:.6f} seg | Memória AGM: {sub_metrics['mst_peak_mem_mb']:.4f} MB")
        print(f"  - Vértices Ímpares: {sub_metrics['num_odd_nodes']}")
        print(f"  - Tempo Matching: {sub_metrics['matching_time_sec']:.6f} seg | Memória Matching: {sub_metrics['matching_peak_mem_mb']:.4f} MB")
        print(f"  - Tempo Circuito Euleriano: {sub_metrics['eulerian_time_sec']:.6f} seg")
        print(f"  - Sequência da Rota: {route[:10]}... (Total: {len(route)} vértices)")

    elif args.algo == "branch_and_bound":
        bb = BranchAndBoundTSP(dist_mat, points)
        (cost, route, metrics), w_time, p_mem = MetricsTracker.run_with_profiling(
            bb.solve, warm_start_strategy=args.warm_start, timeout_sec=args.timeout
        )
        print("RESULTADOS DA EXECUÇÃO:")
        print(f"  - Custo da Rota Encontrada: {cost:.4f}")
        print(f"  - Tempo Total (Wall Clock): {w_time:.4f} seg")
        print(f"  - Pico de Memória (RAM): {p_mem:.4f} MB")
        print(f"  - Nós Expandidos: {metrics['n_expanded']}")
        print(f"  - Nós Podados: {metrics['n_pruned']}")
        print(f"  - Tamanho Máximo da Heap: {metrics['max_heap_size']}")
        print(f"  - Profundidade Máxima: {metrics['max_depth']}")
        print(f"  - Timed Out: {metrics['timed_out']}")
        print(f"  - Sequência da Rota: {route[:10]}... (Total: {len(route)} vértices)")

    print("=" * 50)


if __name__ == "__main__":
    main()