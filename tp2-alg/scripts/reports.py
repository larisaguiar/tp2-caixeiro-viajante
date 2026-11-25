import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

try:
    from sklearn.metrics import r2_score as _sk_r2
except ImportError:
    _sk_r2 = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if _sk_r2 is not None:
        return float(_sk_r2(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_power_loglog(N, Y):
    """Ajusta Y ≈ a * N^b via regressão linear em log-log."""
    mask = (N > 0) & (Y > 0)
    if mask.sum() < 2:
        return None
    b, log_a = np.polyfit(np.log(N[mask]), np.log(Y[mask]), deg=1)
    a = float(np.exp(log_a))
    pred = a * (N[mask] ** b)
    return a, float(b), _r2(Y[mask], pred)


def _fit_exp_semilog(N, Y):
    """Ajusta Y ≈ a * exp(b * N) via regressão linear em semi-log."""
    mask = Y > 0
    if mask.sum() < 2:
        return None
    b, log_a = np.polyfit(N[mask], np.log(Y[mask]), deg=1)
    a = float(np.exp(log_a))
    pred = a * np.exp(b * N[mask])
    return a, float(b), _r2(Y[mask], pred)


def generate_empirical_complexity_analysis(df_raw: pd.DataFrame, plots_dir: str, out_tex: str):
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams["figure.dpi"] = 300
    fit_rows = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for algo, color in [("twice_around_tree", "tab:blue"), ("christofides", "tab:orange")]:
        sub = (
            df_raw[df_raw["algorithm"] == algo]
            .groupby("num_nodes", as_index=False)[["wall_time_sec", "peak_mem_mb"]]
            .mean()
            .sort_values("num_nodes")
        )
        if len(sub) < 2:
            continue

        N = sub["num_nodes"].to_numpy(dtype=float)
        T = sub["wall_time_sec"].to_numpy(dtype=float)
        M = sub["peak_mem_mb"].to_numpy(dtype=float)
        n_grid = np.linspace(N.min(), N.max(), 200)

        res_t = _fit_power_loglog(N, T)
        if res_t:
            a, b, r2 = res_t
            axes[0].scatter(N, T, color=color, label=f"{algo} (dados)")
            axes[0].plot(
                n_grid,
                a * (n_grid ** b),
                "--",
                color=color,
                label=f"{algo}: $N^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
            )
            fit_rows.append((algo, "Tempo Total", f"{a:.2e} N^{{{b:.2f}}}", r2))

        res_m = _fit_power_loglog(N, M)
        if res_m:
            a, b, r2 = res_m
            axes[1].scatter(N, M, color=color, label=f"{algo} (dados)")
            axes[1].plot(
                n_grid,
                a * (n_grid ** b),
                "--",
                color=color,
                label=f"{algo}: $N^{{{b:.2f}}}$ ($R^2={r2:.3f}$)",
            )
            fit_rows.append((algo, "Memória Pico", f"{a:.2e} N^{{{b:.2f}}}", r2))

    axes[0].set_title("Crescimento Empírico do Tempo ($T \\times N$)")
    axes[0].set_xlabel("Número de Vértices ($N$)")
    axes[0].set_ylabel("Tempo (s)")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Crescimento Empírico da Memória ($\\mathrm{RAM} \\times N$)")
    axes[1].set_xlabel("Número de Vértices ($N$)")
    axes[1].set_ylabel("Pico de RAM (MB)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "empirical_complexity_approx.png"), dpi=300)
    plt.close()

    df_bb = df_raw[(df_raw["algorithm"] == "branch_and_bound") & (~df_raw["timed_out"].astype(bool))]
    if not df_bb.empty:
        plt.figure(figsize=(7.5, 4.8))
        for ws, grp in df_bb.groupby("warm_start"):
            sub = grp.groupby("num_nodes", as_index=False)["wall_time_sec"].mean().sort_values("num_nodes")
            N = sub["num_nodes"].to_numpy(dtype=float)
            T = sub["wall_time_sec"].to_numpy(dtype=float)
            plt.scatter(N, T, label=f"{ws} (dados)")
            res_e = _fit_exp_semilog(N, T)
            if res_e:
                a, b, r2 = res_e
                n_grid = np.linspace(N.min(), N.max(), 200)
                plt.plot(
                    n_grid,
                    a * np.exp(b * n_grid),
                    "--",
                    label=f"{ws}: $ae^{{{b:.2f}N}}$ ($R^2={r2:.3f}$)",
                )
                fit_rows.append((f"B&B ({ws})", "Tempo Exato", f"{a:.2e} e^{{{b:.2f}N}}", r2))

        plt.yscale("log")
        plt.title("Complexidade Empírica do Branch-and-Bound (Semi-Log)")
        plt.xlabel("Número de Vértices ($N$)")
        plt.ylabel("Tempo (s) [log]")
        plt.legend(fontsize=7.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "empirical_complexity_bnb.png"), dpi=300)
        plt.close()

    if fit_rows:
        os.makedirs(os.path.dirname(out_tex), exist_ok=True)
        lines = [
            "% Gerado automaticamente por scripts/analyze_and_report.py",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ajustes empíricos de tempo e memória em função de $N$.}",
            "\\label{tab:ajuste_complexidade}",
            "\\begin{tabular}{lllr}",
            "\\hline",
            "Algoritmo & Métrica & Modelo Ajustado & $R^2$ \\\\",
            "\\hline",
        ]
        for algo, metric, eq, r2 in fit_rows:
            lines.append(f"{algo.replace('_', '\\_')} & {metric} & ${eq}$ & {r2:.4f} \\\\")
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Tabela LaTeX de complexidade salva em: {out_tex}")


def generate_latex_table(df_raw: pd.DataFrame, out_tex: str):
    summary = (
        df_raw.groupby(["algorithm", "warm_start"], dropna=False)
        .agg(
            runs=("cost", "count"),
            mean_cost=("cost", "mean"),
            mean_time=("wall_time_sec", "mean"),
            mean_mem=("peak_mem_mb", "mean"),
            mean_expanded=("n_expanded", "mean"),
            mean_heap=("max_heap_size", "mean"),
            timeouts=("timed_out", "sum"),
        )
        .reset_index()
    )

    lines = [
        "% Gerado automaticamente por scripts/analyze_and_report.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Resumo experimental de custo, tempo, memória e busca no TSP.}",
        "\\label{tab:resumo_tsp}",
        "\\begin{tabular}{llrrrrrr}",
        "\\hline",
        "Algoritmo & Warm-Start & Custo Méd. & Tempo (s) & RAM (MB) & Nós Exp. & Heap Máx. & Timeouts \\\\",
        "\\hline",
    ]

    for _, r in summary.iterrows():
        algo = str(r["algorithm"]).replace("_", "\\_")
        ws = str(r["warm_start"]).replace("_", "\\_")
        lines.append(
            f"{algo} & {ws} & {r['mean_cost']:.2f} & {r['mean_time']:.4f} & "
            f"{r['mean_mem']:.2f} & {r['mean_expanded']:.1f} & {r['mean_heap']:.1f} & {int(r['timeouts'])} \\\\"
        )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tabela LaTeX principal salva em: {out_tex}")


def generate_subroutine_latex_table(df_raw: pd.DataFrame, out_tex: str):
    approx_df = df_raw[df_raw["algorithm"].isin(["twice_around_tree", "christofides"])].copy()
    if approx_df.empty:
        return

    for col in ["mst_time_sec", "matching_time_sec"]:
        if col not in approx_df.columns:
            approx_df[col] = 0.0

    approx_df["other_time_sec"] = (
        approx_df["wall_time_sec"] - approx_df["mst_time_sec"] - approx_df["matching_time_sec"]
    ).clip(lower=0.0)

    safe_total = approx_df["wall_time_sec"].replace(0, np.nan)
    approx_df["mst_pct"] = (approx_df["mst_time_sec"] / safe_total * 100.0).fillna(0.0)
    approx_df["match_pct"] = (approx_df["matching_time_sec"] / safe_total * 100.0).fillna(0.0)
    approx_df["other_pct"] = (approx_df["other_time_sec"] / safe_total * 100.0).fillna(0.0)

    sub_summary = (
        approx_df.groupby("algorithm", dropna=False)
        .agg(
            total_s=("wall_time_sec", "mean"),
            mst_s=("mst_time_sec", "mean"),
            mst_pct=("mst_pct", "mean"),
            match_s=("matching_time_sec", "mean"),
            match_pct=("match_pct", "mean"),
            other_s=("other_time_sec", "mean"),
            other_pct=("other_pct", "mean"),
            mem_mb=("peak_mem_mb", "mean"),
        )
        .reset_index()
    )

    lines = [
        "% Gerado automaticamente por scripts/analyze_and_report.py",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Análise de subrotinas (tempo médio, participação \\% e pico de RAM).}",
        "\\label{tab:subrotinas_tsp}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "Algoritmo & Total (s) & AGM (s) & AGM (\\%) & Match (s) & Match (\\%) & Outros (s) & Outros (\\%) & RAM (MB) \\\\",
        "\\hline",
    ]

    for _, r in sub_summary.iterrows():
        algo = str(r["algorithm"]).replace("_", "\\_")
        lines.append(
            f"{algo} & {r['total_s']:.4f} & {r['mst_s']:.4f} & {r['mst_pct']:.1f}\\% & "
            f"{r['match_s']:.4f} & {r['match_pct']:.1f}\\% & {r['other_s']:.4f} & "
            f"{r['other_pct']:.1f}\\% & {r['mem_mb']:.2f} \\\\"
        )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tabela LaTeX de subrotinas salva em: {out_tex}")


def generate_plots(df_raw: pd.DataFrame, df_topo: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams["figure.dpi"] = 300

    df_bb = df_raw[df_raw["algorithm"] == "branch_and_bound"].copy()

    # 1. Comparativo de Warm-Start (Tempo e Nós Expandidos)
    if not df_bb.empty:
        ws_stats = df_bb.groupby("warm_start")[["wall_time_sec", "n_expanded"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ws_stats["wall_time_sec"].plot(kind="bar", ax=axes[0])
        axes[0].set_title("Tempo Médio por Warm-Start (B&B)")
        axes[0].set_ylabel("Tempo (s)")
        axes[0].tick_params(axis="x", rotation=25)

        ws_stats["n_expanded"].plot(kind="bar", ax=axes[1])
        axes[1].set_title("Nós Expandidos Médios por Warm-Start")
        axes[1].set_ylabel("Nós Expandidos")
        axes[1].tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "warmstart_comparison.png"), dpi=300)
        plt.close()

    # 2. Análise de Subrotinas (Tempo Absoluto e Participação % nas Aproximações)
    approx_df = df_raw[df_raw["algorithm"].isin(["twice_around_tree", "christofides"])].copy()
    if not approx_df.empty:
        for col in ["mst_time_sec", "matching_time_sec"]:
            if col not in approx_df.columns:
                approx_df[col] = 0.0
        approx_df["other_time_sec"] = (
            approx_df["wall_time_sec"] - approx_df["mst_time_sec"] - approx_df["matching_time_sec"]
        ).clip(lower=0.0)

        sub_means = approx_df.groupby("algorithm")[["mst_time_sec", "matching_time_sec", "other_time_sec"]].mean()
        sub_pct = sub_means.div(sub_means.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sub_means.plot(kind="bar", stacked=True, ax=axes[0])
        axes[0].set_title("Tempo Médio por Subrotina (s)")
        axes[0].set_ylabel("Segundos")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].legend(["AGM", "Matching", "Outros (DFS/Euler/Short)"], fontsize=8)

        sub_pct.plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_title("Participação Percentual por Subrotina (%)")
        axes[1].set_ylabel("% do Tempo")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].legend(["AGM", "Matching", "Outros"], fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "subroutine_breakdown.png"), dpi=300)
        plt.close()

    # 3. Análise de Memória (Pico de RAM e Tamanho Máximo da Heap)
    if not df_raw.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

        mem_grp = df_raw.groupby(["algorithm", "warm_start"])["peak_mem_mb"].mean().reset_index()
        mem_grp["label"] = mem_grp["algorithm"] + "\n(" + mem_grp["warm_start"] + ")"
        axes[0].barh(mem_grp["label"], mem_grp["peak_mem_mb"])
        axes[0].set_title("Pico Médio de Memória RAM (tracemalloc)")
        axes[0].set_xlabel("Pico de Memória (MB)")
        axes[0].invert_yaxis()

        if not df_bb.empty and "max_heap_size" in df_bb.columns:
            heap_stats = df_bb.groupby("warm_start")["max_heap_size"].mean()
            heap_stats.plot(kind="bar", ax=axes[1])
            axes[1].set_title("Tamanho Máximo Médio da Heap (B&B)")
            axes[1].set_ylabel("Nós na Heap")
            axes[1].tick_params(axis="x", rotation=25)
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "memory_and_heap_analysis.png"), dpi=300)
        plt.close()

    # 4. Aspect Ratio vs. Nós Expandidos (escala log)
    if not df_bb.empty and not df_topo.empty:
        merged = df_bb.merge(df_topo, on="instance", how="inner")
        merged = merged[merged["n_expanded"] > 0]
        if not merged.empty:
            plt.figure(figsize=(7, 4.5))
            for ws, grp in merged.groupby("warm_start"):
                plt.scatter(grp["aspect_ratio"], grp["n_expanded"], label=ws, alpha=0.8)
            plt.yscale("log")
            plt.xlabel("Aspect Ratio")
            plt.ylabel("Nós Expandidos (escala log)")
            plt.title("Razão de Aspecto vs. Expansão de Nós")
            plt.legend(title="Warm-Start", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "aspect_ratio_vs_expanded.png"), dpi=300)
            plt.close()

    # 5. Feature Importance (Random Forest)
    if not df_bb.empty and not df_topo.empty:
        merged = df_bb.merge(df_topo, on="instance", how="inner")
        feature_cols = [
            c
            for c in [
                "num_nodes",
                "aspect_ratio",
                "convex_hull_ratio",
                "nn_cv",
                "spatial_entropy",
                "mst_max_degree",
                "mst_odd_ratio",
            ]
            if c in merged.columns
        ]
        if len(merged) >= 5 and feature_cols:
            try:
                from sklearn.ensemble import RandomForestRegressor

                X = merged[feature_cols].fillna(0.0)
                y = np.log1p(merged["n_expanded"].clip(lower=0))
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)

                order = np.argsort(rf.feature_importances_)
                plt.figure(figsize=(7, 4.2))
                plt.barh([feature_cols[i] for i in order], rf.feature_importances_[order])
                plt.xlabel("Importância Relativa (Gini / Impurity)")
                plt.title("Importância de Features na Expansão do B&B")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=300)
                plt.close()
            except ImportError:
                print("[AVISO] scikit-learn não instalado; pulando feature_importance.png")

    # 6. Dendrograma das Instâncias
    if not df_topo.empty and len(df_topo) >= 2:
        num_cols = [
            c
            for c in [
                "num_nodes",
                "aspect_ratio",
                "convex_hull_ratio",
                "nn_cv",
                "spatial_entropy",
                "mst_max_degree",
                "mst_odd_ratio",
            ]
            if c in df_topo.columns
        ]
        if num_cols:
            mat = df_topo[num_cols].fillna(0.0).to_numpy(dtype=float)
            stds = mat.std(axis=0)
            stds[stds == 0] = 1.0
            mat_norm = (mat - mat.mean(axis=0)) / stds
            Z = linkage(pdist(mat_norm, metric="euclidean"), method="ward")

            plt.figure(figsize=(8, 4.5))
            dendrogram(Z, labels=df_topo["instance"].astype(str).tolist(), leaf_rotation=45)
            plt.title("Dendrograma Topológico das Instâncias (Ward)")
            plt.ylabel("Distância Euclidiana Padronizada")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "dendrogram_instances.png"), dpi=300)
            plt.close()

    print(f"Gráficos salvos em: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analisa resultados e gera tabela LaTeX e gráficos")
    parser.add_argument("--results_dir", type=str, default=os.path.join(ROOT_DIR, "results"))
    args = parser.parse_args()

    raw_path = os.path.join(args.results_dir, "raw_benchmark.csv")
    topo_path = os.path.join(args.results_dir, "topological_features.csv")
    tex_path = os.path.join(args.results_dir, "table_summary.tex")
    sub_tex_path = os.path.join(args.results_dir, "table_subroutines.tex")
    fit_tex_path = os.path.join(args.results_dir, "table_complexity_fit.tex")
    plots_dir = os.path.join(args.results_dir, "plots")

    if not os.path.exists(raw_path):
        print(f"Arquivo não encontrado: {raw_path}")
        sys.exit(1)

    df_raw = pd.read_csv(raw_path)
    df_topo = pd.read_csv(topo_path) if os.path.exists(topo_path) else pd.DataFrame()

    generate_latex_table(df_raw, tex_path)
    generate_subroutine_latex_table(df_raw, sub_tex_path)
    generate_empirical_complexity_analysis(df_raw, plots_dir, fit_tex_path)
    generate_plots(df_raw, df_topo, plots_dir)


if __name__ == "__main__":
    main()