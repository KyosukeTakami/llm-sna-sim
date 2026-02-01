# src/calibrate.py
import argparse
import json
import os
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from metrics import compute_basic_metrics, sfi_distance
from utils import ensure_dir, set_seed
from simulate import run_simulate
from llm_agent import LLMConfig


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def random_search(
    nodes_csv: str,
    edges_csv: str,
    targets_json: str,
    outdir: str,
    rep: int = 20,
    trials: int = 60,
    seed: int = 0,
    T: int = 20,
    llm_provider: str = "dummy",
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.0,
    llm_seed: Optional[int] = None,
    llm_max_tokens: int = 256,
    llm_top_p: float = 1.0,
    fixed_add_if_absent: Optional[int] = None,
    fixed_weight_rule: Optional[str] = None,
    fixed_rank_weights: Optional[str] = None,
    fixed_alpha: Optional[float] = None,
    fixed_top_k: Optional[int] = None,
    fixed_p_new: Optional[float] = None,
    init_graph: str = "observed",   # observed|empty
):
    set_seed(seed)
    ensure_dir(outdir)
    rng = np.random.default_rng(seed)

    targets = _load_json(targets_json)

    weights_path = os.path.join("config", "sfi_weights.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"sfi_weights.json not found: {weights_path}")
    weights = _load_json(weights_path)

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    # ---- 探索レンジ（community無し）----
    ranges = {
        "base_rate": (0.05, 0.30),
        "beta_E": (-1.0, 0.60),
        "beta_friction": (0.0, 0.5),
        "top_k": (2, 5),
        "alpha": (0.2, 3.0),
        "p_new": (0.05, 1.0),
    }

    weight_rule_candidates = ["rank", "score", "rank_score"]
    rank_weights_candidates: List[Optional[str]] = [
        "1.0,0.7,0.5,0.3,0.2",
        "1.0,0.5,0.25,0.2,0.1",
        "1.0,0.8,0.6,0.4,0.2",
        None,
    ]
    add_if_absent_candidates = [0, 1]

    sim_llm_seed = seed if llm_seed is None else llm_seed
    llm_cfg = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        temperature=float(llm_temperature),
        top_p=float(llm_top_p),
        max_tokens=int(llm_max_tokens),
        seed=int(sim_llm_seed),
    )

    recs = []

    for trial in range(trials):
        params: Dict[str, Any] = {
            "base_rate": float(np.round(rng.uniform(*ranges["base_rate"]), 3)),
            "beta_E": float(np.round(rng.uniform(*ranges["beta_E"]), 3)),
            "beta_friction": float(np.round(rng.uniform(*ranges["beta_friction"]), 3)),
        }

        # top_k
        if fixed_top_k is not None:
            params["top_k"] = int(fixed_top_k)
        else:
            params["top_k"] = int(rng.integers(ranges["top_k"][0], ranges["top_k"][1] + 1))

        # weight_rule
        params["weight_rule"] = str(fixed_weight_rule) if fixed_weight_rule is not None else str(rng.choice(weight_rule_candidates))

        # alpha
        params["alpha"] = float(fixed_alpha) if fixed_alpha is not None else float(np.round(rng.uniform(*ranges["alpha"]), 3))

        # add_if_absent
        params["add_if_absent"] = int(fixed_add_if_absent) if fixed_add_if_absent is not None else int(rng.choice(add_if_absent_candidates))

        # rank_weights（None は None のまま）
        if fixed_rank_weights is not None:
            params["rank_weights"] = fixed_rank_weights
        else:
            params["rank_weights"] = rng.choice(rank_weights_candidates)

        # p_new
        if fixed_p_new is not None:
            params["p_new"] = float(fixed_p_new)
        else:
            params["p_new"] = float(np.round(rng.uniform(*ranges["p_new"]), 3))

        losses = []
        for r in range(rep):
            outdir_r = os.path.join(outdir, f"trial{trial}_rep{r}")
            ensure_dir(outdir_r)
            sim_seed = seed + 1000 * trial + r

            sim_edges_path = run_simulate(
                nodes=nodes,
                observed_edges=edges,
                outdir=outdir_r,
                T=int(T),
                params=params,
                seed=int(sim_seed),
                use_persona=True,
                use_climate=True,
                llm_cfg=llm_cfg,
                init_graph=init_graph,
            )
            sim_edges = pd.read_csv(sim_edges_path)
            sim_metrics = compute_basic_metrics(nodes, sim_edges)
            loss = sfi_distance(sim_metrics, targets, weights)
            losses.append(loss)

        rec = {"trial": trial, **params, "loss_mean": float(np.mean(losses)), "loss_sd": float(np.std(losses))}
        recs.append(rec)

    df = pd.DataFrame(recs).sort_values("loss_mean")
    leaderboard_path = os.path.join(outdir, "leaderboard.csv")
    df.to_csv(leaderboard_path, index=False)

    best = df.iloc[0].to_dict()

    best_params = {
        "base_rate": best.get("base_rate"),
        "beta_E": best.get("beta_E"),
        "beta_friction": best.get("beta_friction"),
        "top_k": int(best.get("top_k")),
        "weight_rule": best.get("weight_rule"),
        "rank_weights": best.get("rank_weights"),
        "alpha": best.get("alpha"),
        "add_if_absent": int(best.get("add_if_absent")),
        "p_new": best.get("p_new"),
    }

    best_path = os.path.join(outdir, "best_params.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print(df.head(10).to_string(index=False))
    print("Leaderboard saved ->", leaderboard_path)
    print("Best params saved ->", best_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--rep", type=int, default=20)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=20)

    ap.add_argument("--init_graph", type=str, default="observed")  # observed|empty

    # ---- LLM config ----
    ap.add_argument("--llm.provider", dest="llm_provider", type=str, default="dummy")
    ap.add_argument("--llm.model", dest="llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--llm.temperature", dest="llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm.seed", dest="llm_seed", type=int, default=None)
    ap.add_argument("--llm.max_tokens", dest="llm_max_tokens", type=int, default=256)
    ap.add_argument("--llm.top_p", dest="llm_top_p", type=float, default=1.0)

    # ---- 固定したい params ----
    ap.add_argument("--param.add_if_absent", dest="param_add_if_absent", type=int, default=None)
    ap.add_argument("--param.weight_rule", dest="param_weight_rule", type=str, default=None)
    ap.add_argument("--param.rank_weights", dest="param_rank_weights", type=str, default=None)
    ap.add_argument("--param.alpha", dest="param_alpha", type=float, default=None)
    ap.add_argument("--param.top_k", dest="param_top_k", type=int, default=None)
    ap.add_argument("--param.p_new", dest="param_p_new", type=float, default=None)

    args = ap.parse_args()

    random_search(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        targets_json=args.targets,
        outdir=args.outdir,
        rep=int(args.rep),
        trials=int(args.trials),
        seed=int(args.seed),
        T=int(args.T),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=float(args.llm_temperature),
        llm_seed=args.llm_seed,
        llm_max_tokens=int(args.llm_max_tokens),
        llm_top_p=float(args.llm_top_p),
        fixed_add_if_absent=args.param_add_if_absent,
        fixed_weight_rule=args.param_weight_rule,
        fixed_rank_weights=args.param_rank_weights,
        fixed_alpha=args.param_alpha,
        fixed_top_k=args.param_top_k,
        fixed_p_new=args.param_p_new,
        init_graph=str(args.init_graph),
    )


if __name__ == "__main__":
    main()

