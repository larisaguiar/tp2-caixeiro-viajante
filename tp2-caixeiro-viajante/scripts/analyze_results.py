import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def analyze_results(benchmark_csv: str, features_csv: str, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(benchmark_csv) or not os.path.exists(features_csv):
        print("Arquivos de entrada CSV não encontrados.")
        return

    df_bench = pd.read_csv(benchmark_csv)
    df_feats = pd.read_csv(features_csv)

    # Filtra colunas duplicadas antes de fazer o merge para evitar sufixos _x e _y
    cols_to_use = [c for c in df_feats.columns if c not in df_bench.columns or c == 'instance']
    merged = pd.merge(df_bench, df_feats[cols_to_use], on='instance')
    
    print("=== RESUMO DOS EXPERIMENTOS ===")
    print(f"Total de execuções registradas: {len(df_bench)}")
    print(f"Algoritmos testados: {df_bench['algorithm'].unique()}")

    # 1. Comparativo de Warm-Starts no B&B
    bb_df = df_bench[df_bench['algorithm'] == 'branch_and_bound']
    if not bb_df.empty:
        ws_summary = bb_df.groupby('warm_start_strategy')[['n_expanded', 'n_pruned', 'wall_time_sec', 'peak_mem_mb']].mean()
        print("\n--- Desempenho Médio das Estratégias de Warm-Start (B&B) ---")
        print(ws_summary)

        # Plot 1: Tempo Médio por Warm-Start
        plt.figure(figsize=(8, 5))
        ws_summary['wall_time_sec'].plot(kind='bar', color='#1f77b4', edgecolor='black')
        plt.title('Tempo de Execução Médio do Branch-and-Bound por Warm-Start')
        plt.ylabel('Tempo (segundos)')
        plt.xlabel('Estratégia de Warm-Start')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'warmstart_time_comparison.png'), dpi=200)
        plt.close()

    # 2. Regressão com Random Forest (Prever Nós Expandidos a partir da Topologia)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    feature_cols = ['num_nodes', 'aspect_ratio', 'convex_hull_ratio', 'nn_cv', 'spatial_entropy', 'mst_max_degree', 'mst_odd_ratio']

    if not bb_df.empty and len(merged) > 3:
    # 1. Garante que a coluna n_expanded é numérica (converte 'NA' para NaN)
        merged['n_expanded'] = pd.to_numeric(merged['n_expanded'], errors='coerce')

    # 2. Filtra apenas as linhas que possuem dados válidos em X e y
        clean_df = merged.dropna(subset=feature_cols + ['n_expanded'])

    # 3. Executa o modelo se houver dados limpos suficientes
    if len(clean_df) > 3:
        X = clean_df[feature_cols]
        y = clean_df['n_expanded']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)

        r2 = r2_score(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print(f"\n--- Regressão Random Forest (Previsão de Nós Expandidos no B&B) ---")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nImportância das Características Topológicas:")
        print(importances)

        # Plot 2: Importância de Features
        plt.figure(figsize=(8, 5))
        importances.plot(kind='barh', color='#2ca02c', edgecolor='black')
        plt.title('Importância das Características Topológicas para Poda no B&B')
        plt.xlabel('Importância Relativa')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'topological_feature_importance.png'), dpi=200)
        plt.close()

    print(f"\n Análise estatística concluída! Gráficos salvos em '{plots_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Análise Estatística dos Benchmarks do TP2")
    parser.add_argument("--benchmark_csv", type=str, default="/workspace/scratch/tp2-tsp/results/raw_benchmark.csv")
    parser.add_argument("--features_csv", type=str, default="/workspace/scratch/tp2-tsp/results/topological_features.csv")
    parser.add_argument("--plots_dir", type=str, default="/workspace/scratch/tp2-tsp/plots")
    args = parser.parse_args()

    analyze_results(args.benchmark_csv, args.features_csv, args.plots_dir)