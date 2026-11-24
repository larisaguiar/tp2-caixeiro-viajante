import os
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.parser.tsplib_parser import TSPLibParser
from src.features.topology_extractor import TopologyExtractor
from src.algorithms.approx.twice_around_tree import twice_around_tree_tsp
from src.algorithms.approx.christofides import christofides_tsp
from src.algorithms.exact.branch_and_bound import BranchAndBoundTSP
from src.profiling.metrics_tracker import MetricsTracker

def run_benchmarks(data_dir: str, output_dir: str, timeout_sec: float = 1800.0):
    os.makedirs(output_dir, exist_ok=True)
    raw_csv_path = os.path.join(output_dir, 'raw_benchmark.csv')
    feat_csv_path = os.path.join(output_dir, 'topological_features.csv')

    tsp_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tsp')])
    print(f"Encontrados {len(tsp_files)} arquivos .tsp para benchmarking.\n")

    for tsp_file in tsp_files:
        filepath = os.path.join(data_dir, tsp_file)
        print(f"--- Processando: {tsp_file} ---")

        points, dist_mat, _ = TSPLibParser.parse_file(filepath)
        feat = TopologyExtractor.extract_features(points)
        feat['instance'] = tsp_file

        # Salva as características topológicas incrementalmente
        pd.DataFrame([feat]).to_csv(
            feat_csv_path, mode='a', header=not os.path.exists(feat_csv_path), index=False
        )

        instance_results = []

        # 1. Twice-around-the-tree
        print("  Running Twice-around-the-tree...")
        (cost_tat, route_tat, metrics_tat), time_tat, mem_tat = MetricsTracker.run_with_profiling(
            twice_around_tree_tsp, dist_mat
        )
        instance_results.append({
            'instance': tsp_file, 'algorithm': 'twice_around_tree', 'warm_start_strategy': 'none',
            'cost': cost_tat, 'wall_time_sec': time_tat, 'peak_mem_mb': mem_tat,
            **metrics_tat, **feat
        })

        # 2. Christofides
        print("  Running Christofides...")
        (cost_chris, route_chris, metrics_chris), time_chris, mem_chris = MetricsTracker.run_with_profiling(
            christofides_tsp, dist_mat
        )
        instance_results.append({
            'instance': tsp_file, 'algorithm': 'christofides', 'warm_start_strategy': 'none',
            'cost': cost_chris, 'wall_time_sec': time_chris, 'peak_mem_mb': mem_chris,
            **metrics_chris, **feat
        })

        # 3. Branch-and-Bound (5 Warm-Starts)
        bb_solver = BranchAndBoundTSP(dist_mat, points)
        strategies = ["cold_start", "nearest_neighbor", "convex_hull", "twice_around_tree", "christofides"]

        for ws in strategies:
            print(f"  Running Branch-and-Bound (Warm-Start: {ws})...")
            (cost_bb, route_bb, metrics_bb), time_bb, mem_bb = MetricsTracker.run_with_profiling(
                bb_solver.solve, warm_start_strategy=ws, timeout_sec=timeout_sec
            )
            instance_results.append({
                'instance': tsp_file, 'algorithm': 'branch_and_bound', 'warm_start_strategy': ws,
                'cost': cost_bb if not metrics_bb['timed_out'] else 'NA',
                'wall_time_sec': time_bb, 'peak_mem_mb': mem_bb,
                **metrics_bb, **feat
            })

        # Salva os resultados desta instância imediatamente no CSV
        pd.DataFrame(instance_results).to_csv(
            raw_csv_path, mode='a', header=not os.path.exists(raw_csv_path), index=False
        )
        print(f"  ✅ Instância {tsp_file} salva em {raw_csv_path}!\n")

    # Salvar DataFrames
    df_feats = pd.DataFrame(feature_rows)

    df_feats.to_csv(os.path.join(output_dir, 'topological_features.csv'), index=False)

    print(f"\n Benchmarks concluídos! Resultados salvos em '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automação de Benchmarks de TSP para TP2")
    parser.add_argument("--data_dir", type=str, default="/workspace/scratch/tp2-tsp/data", help="Pasta com instâncias .tsp")
    parser.add_argument("--output_dir", type=str, default="/workspace/scratch/tp2-tsp/results", help="Pasta para resultados CSV")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout do Branch-and-Bound por instância")
    args = parser.parse_args()

    run_benchmarks(args.data_dir, args.output_dir, args.timeout)