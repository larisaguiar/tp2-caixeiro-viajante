import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def generate_all_plots(raw_csv: str, feat_csv: str, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(raw_csv) or not os.path.exists(feat_csv):
        print(f"❌ Arquivos de dados não encontrados em {raw_csv} ou {feat_csv}.")
        return

    # 1. Carrega e unifica os dados
    df_bench = pd.read_csv(raw_csv)
    df_feats = pd.read_csv(feat_csv)
    cols_to_use = [c for c in df_feats.columns if c not in df_bench.columns or c == 'instance']
    merged = pd.merge(df_bench, df_feats[cols_to_use], on='instance')

    # Estilo visual limpo para publicação
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    # -------------------------------------------------------------
    # Gráfico 1: Comparativo de Tempo e Podas por Warm-Start
    # -------------------------------------------------------------
    bb_df = merged[merged['algorithm'] == 'branch_and_bound'].copy()
    bb_df['wall_time_sec'] = pd.to_numeric(bb_df['wall_time_sec'], errors='coerce')
    bb_df['n_pruned'] = pd.to_numeric(bb_df['n_pruned'], errors='coerce')

    ws_summary = bb_df.groupby('warm_start_strategy').agg({
        'wall_time_sec': 'mean',
        'n_pruned': 'mean'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=300)
    color1 = '#1f77b4'
    ax1.set_xlabel('Estratégia de Warm-Start', fontweight='bold')
    ax1.set_ylabel('Tempo Médio (s)', color=color1, fontweight='bold')
    ax1.bar(ws_summary['warm_start_strategy'], ws_summary['wall_time_sec'], color=color1, alpha=0.75, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color1)
    plt.xticks(rotation=15)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Média de Nós Podados', color=color2, fontweight='bold')
    ax2.plot(ws_summary['warm_start_strategy'], ws_summary['n_pruned'], color=color2, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Impacto do Warm-Start no Tempo e na Poda do B&B', fontweight='bold', pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'warmstart_comparison.png'))
    plt.close()

    # -------------------------------------------------------------
    # Gráfico 2: Importância das Características Topológicas (Random Forest)
    # -------------------------------------------------------------
    feature_cols = ['num_nodes', 'aspect_ratio', 'convex_hull_ratio', 'nn_cv', 'spatial_entropy', 'mst_max_degree', 'mst_odd_ratio']
    clean_rf = merged.dropna(subset=feature_cols + ['n_expanded']).copy()
    clean_rf['n_expanded'] = pd.to_numeric(clean_rf['n_expanded'], errors='coerce')
    clean_rf = clean_rf.dropna(subset=['n_expanded'])

    if len(clean_rf) > 3:
        X = clean_rf[feature_cols]
        y = clean_rf['n_expanded']
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

        plt.figure(figsize=(6.5, 3.8), dpi=300)
        importances.plot(kind='barh', color='#2ca02c', edgecolor='#1b631b')
        plt.title('Importância das Características na Regressão (Random Forest)', fontweight='bold')
        plt.xlabel('Importância Relativa', fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        plt.close()

    # -------------------------------------------------------------
    # Gráfico 3: Aspect Ratio vs Nós Expandidos (Escala Log)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 4), dpi=300)
    sns.scatterplot(data=clean_rf, x='aspect_ratio', y='n_expanded', hue='warm_start_strategy', s=90, palette='Set1')
    plt.yscale('log')
    plt.title('Razão de Aspecto vs. Nós Expandidos no B&B', fontweight='bold')
    plt.xlabel('Razão de Aspecto (Aspect Ratio)', fontweight='bold')
    plt.ylabel('Nós Expandidos (Escala Logarítmica)', fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(title='Warm-Start', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'aspect_ratio_vs_expanded.png'))
    plt.close()

    print(f"✅ Todos os gráficos foram salvos com sucesso na pasta '{plots_dir}/'!")

if __name__ == '__main__':
    generate_all_plots('results/raw_benchmark.csv', 'results/topological_features.csv', 'plots')