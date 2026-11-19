import random
from pathlib import Path


def generate_tsp_euc2d(
    output_dir: Path,
    name: str,
    n: int,
    seed: int,
    coord_max: float = 1000.0,
) -> Path:
    if not (10 <= n <= 75):
        raise ValueError(f"N={n} fora da faixa permitida [10, 75].")

    rng = random.Random(seed)
    coords = [
        (
            round(rng.uniform(0.0, coord_max), 2),
            round(rng.uniform(0.0, coord_max), 2),
        )
        for _ in range(n)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.tsp"

    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"COMMENT: Synthetic EUC_2D instance (N={n}, seed={seed})",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]

    for idx, (x, y) in enumerate(coords, start=1):
        lines.append(f"{idx} {x:.2f} {y:.2f}")

    lines.append("EOF")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filepath


def main() -> None:
    data_dir = Path("data")
    # Exemplo: 10 instâncias na faixa N <= 40 para compor 80% de um conjunto de 12 instâncias
    # (combinadas com 2 instâncias oficiais EUC_2D na faixa 45 <= N <= 75, ex.: eil51 e berlin52)
    targets = [
        ("synth_n10_s101", 10, 101),
        ("synth_n14_s102", 14, 102),
        ("synth_n18_s103", 18, 103),
        ("synth_n22_s104", 22, 104),
        ("synth_n25_s105", 25, 105),
        ("synth_n28_s106", 28, 106),
        ("synth_n32_s107", 32, 107),
        ("synth_n35_s108", 35, 108),
        ("synth_n38_s109", 38, 109),
        ("synth_n40_s110", 40, 110),
    ]

    for name, n, seed in targets:
        path = generate_tsp_euc2d(data_dir, name, n, seed)
        print(f"Gerado: {path}")


if __name__ == "__main__":
    main()