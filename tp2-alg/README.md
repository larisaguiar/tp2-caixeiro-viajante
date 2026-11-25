# TP2 - Caixeiro Viajante (TSP)

Projeto para executar algoritmos de aproximação e Branch-and-Bound para o
Problema do Caixeiro Viajante usando instâncias `.tsp` com coordenadas `EUC_2D`.

## Requisitos

Python e `pip`. Na pasta `tp2-alg`, crie um ambiente virtual e instale as
dependências:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows, ative o ambiente com:

```bat
.venv\Scripts\activate
```

## Executar um algoritmo

Execute os comandos a partir da pasta `tp2-alg`. Por exemplo, para executar
Christofides em uma instância incluída no projeto:

```bash
python main.py --instance data/berlin52.tsp --algo christofides
```

Algoritmos disponíveis: `twice_around_tree`, `christofides` e
`branch_and_bound`. Para Branch-and-Bound, também é possível escolher a
estratégia de warm-start e o limite de tempo:

```bash
python main.py --instance data/berlin52.tsp --algo branch_and_bound \
  --warm_start nearest_neighbor --timeout 60
```

Warm-starts disponíveis: `cold_start`, `nearest_neighbor`, `convex_hull`,
`twice_around_tree` e `christofides`.

## Executar benchmarks e gerar relatórios

Para executar os benchmarks em todas as instâncias de `data/`:

```bash
python scripts/run_benchmarks.py
```

Os resultados são salvos em `results/`. Para analisar os CSVs e gerar tabelas
LaTeX e gráficos:

```bash
python scripts/reports.py
```

