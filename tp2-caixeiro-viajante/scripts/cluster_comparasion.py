
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

FEATURES = [
    'num_nodes',
    'aspect_ratio',
    'convex_hull_ratio',
    'spatial_entropy',
    'mst_odd_ratio'
]

CUT_HEIGHT = 3.5

FEATURES_NO_NODES = [
    'aspect_ratio',
    'convex_hull_ratio',
    'spatial_entropy',
    'mst_odd_ratio'
]

FEATURES_FILE = 'results/topological_features.csv'
RESULTS_FILE = 'results/raw_benchmark.csv'

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

df_feats = pd.read_csv(FEATURES_FILE)

df_feats = df_feats.dropna(
    subset=FEATURES
).copy()

print('\nINSTÂNCIAS ANALISADAS:')
print(df_feats['instance'].tolist())

print(f'\nTotal de instâncias: {len(df_feats)}')


# ============================================================
# 3. AGRUPAMENTO COM TODAS AS FEATURES
# ============================================================

scaler_full = StandardScaler()

X_full = scaler_full.fit_transform(
    df_feats[FEATURES]
)

Z_full = linkage(
    X_full,
    method='ward'
)

df_feats['cluster'] = fcluster(
    Z_full,
    t=CUT_HEIGHT,
    criterion='distance'
)


# ============================================================
# 4. AGRUPAMENTO SEM NUM_NODES
# ============================================================

scaler_reduced = StandardScaler()

X_reduced = scaler_reduced.fit_transform(
    df_feats[FEATURES_NO_NODES]
)

Z_reduced = linkage(
    X_reduced,
    method='ward'
)

df_feats['cluster_no_nodes'] = fcluster(
    Z_reduced,
    t=CUT_HEIGHT,
    criterion='distance'
)


# ============================================================
# 5. COMPARAÇÃO DOS DENDROGRAMAS
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(17, 6)
)

dendrogram(
    Z_full,
    labels=df_feats['instance'].values,
    leaf_rotation=45,
    leaf_font_size=8,
    color_threshold=CUT_HEIGHT,
    ax=axes[0]
)

axes[0].set_title(
    'Com todas as features',
    fontweight='bold'
)

axes[0].set_ylabel('Distância de Ward')
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

dendrogram(
    Z_reduced,
    labels=df_feats['instance'].values,
    leaf_rotation=45,
    leaf_font_size=8,
    color_threshold=CUT_HEIGHT,
    ax=axes[1]
)

axes[1].set_title(
    'Sem num_nodes',
    fontweight='bold'
)

axes[1].set_ylabel('Distância de Ward')
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle(
    'Comparação dos agrupamentos hierárquicos TSP',
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    'plots/comparacao_dendrogramas.png',
    dpi=300
)

plt.close()


# ============================================================
# 6. CARACTERIZAÇÃO DOS CLUSTERS
# ============================================================

cluster_means = df_feats.groupby(
    'cluster'
)[FEATURES].mean()

cluster_std = df_feats.groupby(
    'cluster'
)[FEATURES].std()

cluster_counts = df_feats.groupby(
    'cluster'
)['instance'].count()

print('\n========================================')
print('CARACTERIZAÇÃO DOS CLUSTERS')
print('========================================')

print('\nMédias:')
print(cluster_means.round(3))

print('\nDesvios padrão:')
print(cluster_std.round(3))

print('\nQuantidade de instâncias:')
print(cluster_counts)

cluster_means.to_csv(
    'results/cluster_means.csv'
)

cluster_std.to_csv(
    'results/cluster_std.csv'
)


# ============================================================
# 7. VISUALIZAÇÃO DAS FEATURES POR CLUSTER
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 8)
)

axes = axes.flatten()

for i, feature in enumerate(FEATURES):

    sns.boxplot(
        data=df_feats,
        x='cluster',
        y=feature,
        ax=axes[i]
    )

    axes[i].set_title(feature)
    axes[i].set_xlabel('Cluster')

axes[-1].axis('off')

plt.suptitle(
    'Distribuição das características por cluster',
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    'plots/features_por_cluster.png',
    dpi=300
)

plt.close()


# ============================================================
# 8. COMPARAÇÃO DOS CLUSTERS
# ============================================================

comparison = df_feats[
    ['instance', 'cluster', 'cluster_no_nodes']
].copy()

print('\n========================================')
print('COMPARAÇÃO DOS AGRUPAMENTOS')
print('========================================')

print(comparison.to_string(index=False))

comparison.to_csv(
    'results/comparacao_clusters.csv',
    index=False
)


# ============================================================
# 9. ANÁLISE DE DESEMPENHO DOS ALGORITMOS
# ============================================================
# ============================================================
# ANÁLISE DE DESEMPENHO
# ============================================================

df_results = pd.read_csv('results/raw_benchmark.csv')

df_results.columns = df_results.columns.str.strip()

# Identificação do resultado
df_results['status'] = df_results['timed_out'].apply(
    lambda x: 'Estouro (Timeout)' if x else 'Concluído'
)

# Relaciona características das instâncias aos resultados
df = df_results.merge(
    df_feats,
    on='instance',
    how='left',
    suffixes=('', '_features')
)

# Resumo por cluster e algoritmo
performance = df.groupby(
    ['cluster', 'algorithm']
).agg(
    total=('instance', 'count'),
    concluidos=('timed_out', lambda x: (~x.astype(bool)).sum()),
    timeouts=('timed_out', lambda x: x.astype(bool).sum()),
    tempo_medio=('wall_time_sec', 'mean'),
    nos_expandidos=('n_expanded', 'mean'),
    custo_medio=('cost', 'mean')
).reset_index()

print('\nDESEMPENHO POR CLUSTER E ALGORITMO:')
print(performance.to_string(index=False))

performance.to_csv(
    'results/performance_by_cluster.csv',
    index=False
)
# ============================================================
# 10. EXPORTAÇÃO FINAL
# ============================================================

df_feats.to_csv(
    'results/instances_with_clusters.csv',
    index=False
)

print('\n========================================')
print('ANÁLISE CONCLUÍDA!')
print('========================================')

print('\nArquivos gerados:')

print('- plots/comparacao_dendrogramas.png')
print('- plots/features_por_cluster.png')
print('- plots/cluster_performance.png')
print('- results/cluster_means.csv')
print('- results/cluster_std.csv')
print('- results/comparacao_clusters.csv')
print('- results/cluster_performance.csv')
print('- results/instances_with_clusters.csv')
