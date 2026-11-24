import os
import pandas as pd

def generate_latex_table(raw_csv: str, feat_csv: str, output_tex: str):
    if not os.path.exists(raw_csv) or not os.path.exists(feat_csv):
        print(f"❌ Erro: Arquivos CSV não encontrados em {raw_csv} ou {feat_csv}.")
        return

    # 1. Carrega os CSVs
    df_bench = pd.read_csv(raw_csv)
    df_feats = pd.read_csv(feat_csv)

    # 2. Une os dados
    cols_to_use = [c for c in df_feats.columns if c not in df_bench.columns or c == 'instance']
    merged = pd.merge(df_bench, df_feats[cols_to_use], on='instance')

    # Helper para extrair valores com segurança
    def get_val(df_sub, col):
        if df_sub.empty:
            return None
        val = df_sub[col].iloc[0]
        val_num = pd.to_numeric(val, errors='coerce')
        return None if pd.isna(val_num) else float(val_num)

    # 3. Processa cada instância
    summary_rows = []
    for instance, group in merged.groupby('instance'):
        n_nodes = group['num_nodes'].iloc[0]
        aspect_ratio = group['aspect_ratio'].iloc[0]
        
        # Filtra as estratégias de Branch-and-Bound
        bb_cold = group[(group['algorithm'] == 'branch_and_bound') & (group['warm_start_strategy'] == 'cold_start')]
        bb_tat = group[(group['algorithm'] == 'branch_and_bound') & (group['warm_start_strategy'] == 'twice_around_tree')]
        bb_chris = group[(group['algorithm'] == 'branch_and_bound') & (group['warm_start_strategy'] == 'christofides')]
        
        # Heurística Twice-around-the-tree isolada
        approx_tat = group[group['algorithm'] == 'twice_around_tree']
        
        cost_opt = get_val(bb_chris, 'cost')
        cost_tat = get_val(approx_tat, 'cost')
        
        time_cold = get_val(bb_cold, 'wall_time_sec')
        time_tat = get_val(bb_tat, 'wall_time_sec')
        time_chris = get_val(bb_chris, 'wall_time_sec')
        
        n_exp_tat = get_val(bb_tat, 'n_expanded')
        n_exp_chris = get_val(bb_chris, 'n_expanded')
        
        timed_out = bb_chris['timed_out'].iloc[0] if not bb_chris.empty and 'timed_out' in bb_chris.columns else False
        status = "Ótimo" if (cost_opt is not None and not timed_out) else "Estouro (OOM/Timeout)"

        summary_rows.append({
            'Instância': f"\\texttt{{{instance}}}",
            'N': int(n_nodes) if pd.notnull(n_nodes) else '-',
            'Aspect Ratio': f"{aspect_ratio:.2f}" if pd.notnull(aspect_ratio) else '-',
            'Custo Ótimo': f"{cost_opt:.2f}" if cost_opt is not None else 'NA',
            'Tempo Cold (s)': f"{time_cold:.2f}" if time_cold is not None else 'NA',
            'Tempo Twice (s)': f"{time_tat:.2f}" if time_tat is not None else 'NA',
            'Tempo Chris (s)': f"{time_chris:.2f}" if time_chris is not None else 'NA',
            'Nós Exp. (Twice)': f"{int(n_exp_tat):,}".replace(",", ".") if n_exp_tat is not None else '>1.000.000',
            'Nós Exp. (Chris)': f"{int(n_exp_chris):,}".replace(",", ".") if n_exp_chris is not None else '>1.000.000',
            'Status': status
        })

    df_table = pd.DataFrame(summary_rows)

    # 4. Gera código LaTeX
    latex_code = df_table.to_latex(
        index=False,
        escape=False,
        caption="Desempenho comparativo detalhado incluindo Twice-around-the-tree e Christofides.",
        label="tab:resultados_instancias",
        column_format="lccccccccr"
    )

    with open(output_tex, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    print(f"✅ Tabela completa atualizada em '{output_tex}'!")

if __name__ == '__main__':
    generate_latex_table('results/raw_benchmark.csv', 'results/topological_features.csv', 'results/table_summary.tex')