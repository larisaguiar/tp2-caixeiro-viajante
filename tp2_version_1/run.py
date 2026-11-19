import subprocess, glob, csv, os

ALGOS = ["twice_around", "christofides", "bnb_cold", "bnb_nn", "bnb_hull", "bnb_twice", "bnb_christofides"]
HEADER = ["instance", "n", "algo", "status", "cost", "init_ub", "time_ms",
          "time_warmup_ms", "time_mst_ms", "time_match_ms", "time_bnb_ms", "expanded", "pruned", "mem_kb"]

subprocess.run(["make"], check=True)
os.makedirs("results", exist_ok=True)

with open("results/metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    for tsp in sorted(glob.glob("data/*.tsp")):
        for algo in ALGOS:
            print(f"Rodando {os.path.basename(tsp)} -> {algo}")
            out = subprocess.check_output(["./tsp", tsp, algo, "1800"], text=True).strip()
            w.writerow(out.split(","))
            f.flush()
