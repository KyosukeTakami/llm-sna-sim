import argparse, json, pandas as pd
from metrics import compute_basic_metrics
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    m = compute_basic_metrics(nodes, edges)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"Saved targets -> {args.out}")
