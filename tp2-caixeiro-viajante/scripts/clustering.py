import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Carrega as características topológicas
df_feats = pd.read_csv('results/topological_features.csv')

# 2. Seleciona as colunas geométricas para o agrupamento
feature_cols = ['num_nodes', 'aspect_ratio', 'convex_hull_ratio', 'spatial_entropy', 'mst_odd_ratio']
df_clean = df_feats.dropna(subset=feature_cols).copy()

# 3. Normaliza os dados (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[feature_cols])

# 4. Calcula o Clustering Hierárquico (Método de Ward)
Z = linkage(X_scaled, method='ward')

# 5. Plota o Dendrograma em alta resolução
plt.figure(figsize=(9, 4.5), dpi=300)
dendrogram(
    Z,
    labels=df_clean['instance'].values,
    leaf_rotation=45,
    leaf_font_size=10,
    color_threshold=3.5 # Define a linha de corte dos clusters
)

plt.title('Dendrograma de Agrupamento Hierárquico das Instâncias (TSP)', fontweight='bold', pad=12)
plt.ylabel('Distância Euclidiana (Espaço de Features)', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/dendrogram_instances.png')
plt.close()

print("✅ Dendrograma salvo com sucesso em 'plots/dendrogram_instances.png'!")