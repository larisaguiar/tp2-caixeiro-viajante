import argparse
import glob
import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.tsplib_parser import TSPLibParser
from src.topology_extractor import TopologyExtractor
from src.twice_around_tree import twice_around_tree_tsp
from src.christofides import christofides_tsp
from src.branch_bound import BranchAndBoundTSP
from src.metrics import MetricsTracker


def append_row_csv(filepath: str, row: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_row = pd.DataFrame([row])
    header = not os.path.exists(filepath)
    df_row.to_csv(filepath, mode="a", header=header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Executa bateria de benchmarks do TP2 (TSP)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT_DIR, "data"), help="Pasta com arquivos .tsp")
    parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results"), help="Pasta de saída")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout por execução do Branch-and-Bound (seg)")
    parser.add_argument(
        "--warm_starts",
        nargs="+",
        default=["cold_start", "nearest_neighbor", "convex_hull", "twice_around_tree", "christofides"],
        help="Lista de warm-starts para testar no Branch-and-Bound",
    )
    args = parser.parse_args()

    tsp_files = sorted(glob.glob(os.path.join(args.data_dir, "*.tsp")))
    if not tsp_files:
        print(f"Nenhum arquivo .tsp encontrado em: {args.data_dir}")
        sys.exit(1)

    topo_csv = os.path.join(args.results_dir, "topological_features.csv")
    raw_csv = os.path.join(args.results_dir, "raw_benchmark.csv")

    seen_instances = set()
    if os.path.exists(topo_csv):
        try:
            seen_instances = set(pd.read_csv(topo_csv)["instance"].dropna().unique())
        except Exception:
            seen_instances = set()

    for filepath in tsp_files:
        inst_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n>>> Instância: {inst_name} ({filepath})")

        try:
            points, dist_mat, _ = TSPLibParser.parse_file(filepath)
        except Exception as e:
            print(f"  [ERRO] Falha ao ler {filepath}: {e}")
            continue

        if inst_name not in seen_instances:
            try:
                feats = TopologyExtractor.extract_features(points)
                topo_row = {"instance": inst_name, **feats}
                append_row_csv(topo_csv, topo_row)
                seen_instances.add(inst_name)
            except Exception as e:
                print(f"  [AVISO] Falha ao extrair topologia de {inst_name}: {e}")

        # 1. Twice-Around-the-Tree
        (tat_cost, _, tat_sub), tat_time, tat_mem = MetricsTracker.run_with_profiling(
            twice_around_tree_tsp, dist_mat
        )
        append_row_csv(
            raw_csv,
            {
                "instance": inst_name,
                "num_nodes": len(points),
                "algorithm": "twice_around_tree",
                "warm_start": "none",
                "cost": tat_cost,
                "wall_time_sec": tat_time,
                "peak_mem_mb": tat_mem,
                "mst_time_sec": tat_sub["mst_time_sec"],
                "matching_time_sec": 0.0,
                "n_expanded": 0,
                "n_pruned": 0,
                "max_heap_size": 0,
                "max_depth": 0,
                "timed_out": False,
            },
        )
        print(f"  twice_around_tree | custo={tat_cost:.2f} | tempo={tat_time:.4f}s | ram={tat_mem:.2f}MB")

        # 2. Christofides
        (chr_cost, _, chr_sub), chr_time, chr_mem = MetricsTracker.run_with_profiling(
            christofides_tsp, dist_mat
        )
        append_row_csv(
            raw_csv,
            {
                "instance": inst_name,
                "num_nodes": len(points),
                "algorithm": "christofides",
                "warm_start": "none",
                "cost": chr_cost,
                "wall_time_sec": chr_time,
                "peak_mem_mb": chr_mem,
                "mst_time_sec": chr_sub["mst_time_sec"],
                "matching_time_sec": chr_sub["matching_time_sec"],
                "n_expanded": 0,
                "n_pruned": 0,
                "max_heap_size": 0,
                "max_depth": 0,
                "timed_out": False,
            },
        )
        print(f"  christofides      | custo={chr_cost:.2f} | tempo={chr_time:.4f}s | ram={chr_mem:.2f}MB")

        # 3. Branch-and-Bound por estratégia de Warm-Start
        for ws in args.warm_starts:
            bb = BranchAndBoundTSP(dist_mat, points)
            (bb_cost, _, bb_metrics), bb_time, bb_mem = MetricsTracker.run_with_profiling(
                bb.solve, warm_start_strategy=ws, timeout_sec=args.timeout
            )
            append_row_csv(
                raw_csv,
                {
                    "instance": inst_name,
                    "num_nodes": len(points),
                    "algorithm": "branch_and_bound",
                    "warm_start": ws,
                    "cost": bb_cost,
                    "wall_time_sec": bb_time,
                    "peak_mem_mb": bb_mem,
                    "mst_time_sec": 0.0,
                    "matching_time_sec": 0.0,
                    "n_expanded": bb_metrics["n_expanded"],
                    "n_pruned": bb_metrics["n_pruned"],
                    "max_heap_size": bb_metrics["max_heap_size"],
                    "max_depth": bb_metrics["max_depth"],
                    "timed_out": bb_metrics["timed_out"],
                },
            )
            print(
                f"  bnb ({ws:17s}) | custo={bb_cost:.2f} | tempo={bb_time:.4f}s | "
                f"exp={bb_metrics['n_expanded']} | pod={bb_metrics['n_pruned']} | to={bb_metrics['timed_out']}"
            )


if __name__ == "__main__":
    main()