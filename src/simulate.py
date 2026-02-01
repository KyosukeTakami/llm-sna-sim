# src/simulate.py
import argparse
import os
import json
import random
from typing import Dict, Optional, Any, List

import pandas as pd
import networkx as nx

from metrics import compute_basic_metrics
from utils import ensure_dir, set_seed
from llm_agent import LLMConfig, build_llm


# ======== 追加: LLMに渡したい主要特徴量（列名の期待値）========
REQUIRED_TRAIT_KEYS = [
    # Big Five (z)
    "BFI_E_z", "BFI_A_z", "BFI_C_z", "BFI_N_z", "BFI_O_z",
    # Classroom climate (z)
    "CCI_friction_z", "CCI_satisfaction_z", "CCI_closeness_z",
]


def _safe_get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    dictから数値を安全に取り出す。
    - keyが無い / None / NaN / 文字列NaN のとき default
    """
    v = d.get(key, None)
    if v is None:
        return float(default)
    try:
        vf = float(v)
        if pd.isna(vf):
            return float(default)
        return float(vf)
    except Exception:
        return float(default)


def _parse_rank_weights(s: Optional[str], top_k: int) -> List[float]:
    # ここが重要：None や "None" を安全に扱う
    if s is None:
        return [1.0 / (i + 1) for i in range(top_k)]
    if isinstance(s, str) and s.strip().lower() in ("none", ""):
        return [1.0 / (i + 1) for i in range(top_k)]

    ws = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
    if len(ws) == 0:
        return [1.0 / (i + 1) for i in range(top_k)]
    if len(ws) < top_k:
        ws = ws + [ws[-1]] * (top_k - len(ws))
    return ws[:top_k]


def _update_edge_weight(G: nx.Graph, i: str, j: str, delta: float, add_if_absent: bool = True):
    a, b = sorted([i, j])
    if G.has_edge(a, b):
        G[a][b]["weight"] = float(G[a][b].get("weight", 0.0)) + float(delta)
    else:
        if add_if_absent:
            G.add_edge(a, b, weight=float(delta))


def run_simulate(
    nodes: pd.DataFrame,
    observed_edges: pd.DataFrame,
    outdir: str,
    T: int,
    params: Dict[str, Any],
    seed: int = 123,
    use_persona: bool = True,
    use_climate: bool = True,
    llm_cfg: Optional[LLMConfig] = None,
    init_graph: str = "observed",  # observed|empty
) -> str:
    """
    community削除版：
    - 重要：agent/LLMに community/cand_community を渡さない
    - 各iは top_k 人を選ぶ
    - weight_rule に従って重み更新
    - init_graph:
        - observed: 観測辺を初期として入れる
        - empty: 空グラフから生成（構造も生成対象にしたいとき）
    """
    set_seed(seed)
    ensure_dir(outdir)

    # ---- build graph (undirected) ----
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["student_id"], **r.to_dict())

    if init_graph not in ("observed", "empty"):
        raise ValueError(f"Unknown init_graph: {init_graph}")

    if init_graph == "observed":
        for _, r in observed_edges.iterrows():
            u = r["source"]
            v = r["target"]
            w = float(r.get("weight", 1.0))
            a, b = sorted([u, v])
            if G.has_edge(a, b):
                G[a][b]["weight"] = float(G[a][b].get("weight", 0.0)) + w
            else:
                G.add_edge(a, b, weight=w)

    # ---- LLM ----
    if llm_cfg is None:
        llm_cfg = LLMConfig(provider="dummy", seed=seed)
    llm = build_llm(llm_cfg)

    ids = nodes["student_id"].tolist()

    top_k = int(params.get("top_k", 3))

    # ---- weight update rule ----
    weight_rule = str(params.get("weight_rule", "rank")).lower()  # rank|score|rank_score
    alpha = float(params.get("alpha", 1.0))
    add_if_absent = bool(int(params.get("add_if_absent", 1)))
    p_new = float(params.get("p_new", 1.0))  # 「新規エッジ生成」の確率ゲート（任意）

    rank_weights = _parse_rank_weights(params.get("rank_weights", None), top_k)

    logs = []

    # ---- simulate ----
    for t in range(1, T + 1):
        random.shuffle(ids)
        for i in ids:
            row = nodes.loc[nodes["student_id"] == i].iloc[0].to_dict()

            # ★ communityを入力から除去（rowには入っていてもOK）
            row.pop("community", None)

            candidates = [j for j in ids if j != i]

            ctx = dict(row)
            ctx.update(params)

            # ======== 追加: BigFive/CCIを「必ず」「floatで」「NaNなし」で載せる ========
            # traits.csv に列が無い場合も default=0.0 で埋まる（OpenAI payload側が安定）
            for k in REQUIRED_TRAIT_KEYS:
                ctx[k] = _safe_get_float(row, k, default=0.0)

            # ★ cand_community を渡さない
            ctx.update(
                {
                    "use_persona": use_persona,
                    "use_climate": use_climate,
                    "rng_seed": int(seed + 100000 * t + hash(str(i)) % 10000),
                "llm_include_params": False,
                }
            )

            

            ctx.pop("llm_fallback", None)

            scores = llm.score_candidates(ctx, candidates, top_k)
            llm_fallback = bool(ctx.get("llm_fallback", False))

            items = list(scores.items())[:top_k]

            for rank_idx, (j, s) in enumerate(items):
                s = float(s)
                a_r = float(rank_weights[rank_idx]) if rank_idx < len(rank_weights) else float(rank_weights[-1])

                if weight_rule == "rank":
                    delta = alpha * a_r
                elif weight_rule == "score":
                    delta = alpha * s
                elif weight_rule in ("rank_score", "rank*score"):
                    delta = alpha * a_r * s
                else:
                    raise ValueError(f"Unknown weight_rule: {weight_rule}")

                # --- p_new gate（新規エッジだけ抑制したい場合）
                is_new = not G.has_edge(*sorted([i, j]))
                if is_new and add_if_absent:
                    # 新規エッジ生成を確率で抑える
                    # 再現性：rng_seedから乱数を作る
                    rnd = random.Random(int(ctx["rng_seed"]) + 17 * (rank_idx + 1))
                    if rnd.random() > p_new:
                        # ログだけ残してスキップ
                        logs.append(
                            {
                                "t": t,
                                "source": i,
                                "target": j,
                                "rank": int(rank_idx + 1),
                                "score": s,
                                "delta_w": 0.0,
                                "skipped_new_edge": True,
                                "weight_rule": weight_rule,
                                "alpha": alpha,
                                "rank_weight": a_r,
                                "top_k": top_k,
                                "p_new": p_new,
                                "llm_provider": llm_cfg.provider,
                                "llm_model": llm_cfg.model,
                                "llm_temperature": float(llm_cfg.temperature),
                                "llm_fallback": llm_fallback,
                                "init_graph": init_graph,
                            }
                        )
                        continue

                logs.append(
                    {
                        "t": t,
                        "source": i,
                        "target": j,
                        "rank": int(rank_idx + 1),
                        "score": s,
                        "delta_w": float(delta),
                        "skipped_new_edge": False,
                        "weight_rule": weight_rule,
                        "alpha": alpha,
                        "rank_weight": a_r,
                        "top_k": top_k,
                        "p_new": p_new,
                        "llm_provider": llm_cfg.provider,
                        "llm_model": llm_cfg.model,
                        "llm_temperature": float(llm_cfg.temperature),
                        "llm_fallback": llm_fallback,
                        "init_graph": init_graph,
                    }
                )

                _update_edge_weight(G, i, j, delta=delta, add_if_absent=add_if_absent)

    # ---- outputs ----
    edges_out = os.path.join(outdir, "edges_sim.csv")
    pd.DataFrame(
        [{"source": u, "target": v, "weight": d.get("weight", 0.0)} for u, v, d in G.edges(data=True)]
    ).to_csv(edges_out, index=False)

    log_path = os.path.join(outdir, "interaction_log.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)

    return edges_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)

    # init graph
    ap.add_argument("--init_graph", type=str, default="observed")  # observed|empty

    # --- params (interaction model / selection) ---
    ap.add_argument("--param.base_rate", dest="param_base_rate", type=float, default=0.15)
    ap.add_argument("--param.beta_E", dest="param_beta_E", type=float, default=0.15)
    ap.add_argument("--param.beta_friction", dest="param_beta_friction", type=float, default=0.2)
    ap.add_argument("--param.top_k", dest="param_top_k", type=int, default=3)

    # --- params (weight update) ---
    ap.add_argument("--param.weight_rule", dest="param_weight_rule", type=str, default="rank")  # rank|score|rank_score
    ap.add_argument("--param.rank_weights", dest="param_rank_weights", type=str, default=None)
    ap.add_argument("--param.alpha", dest="param_alpha", type=float, default=1.0)
    ap.add_argument("--param.add_if_absent", dest="param_add_if_absent", type=int, default=1)
    ap.add_argument("--param.p_new", dest="param_p_new", type=float, default=1.0)

    # --- ablation ---
    ap.add_argument("--ablate.persona", dest="ablate_persona", type=int, default=0)
    ap.add_argument("--ablate.climate", dest="ablate_climate", type=int, default=0)

    # --- best params ---
    ap.add_argument("--use_best", type=str, default=None)

    # --- LLM config ---
    ap.add_argument("--llm.provider", dest="llm_provider", type=str, default="dummy")  # dummy|openai
    ap.add_argument("--llm.model", dest="llm_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--llm.temperature", dest="llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm.seed", dest="llm_seed", type=int, default=None)
    ap.add_argument("--llm.max_tokens", dest="llm_max_tokens", type=int, default=256)
    ap.add_argument("--llm.top_p", dest="llm_top_p", type=float, default=1.0)

    ap.add_argument("--param.llm_include_params", dest="param_llm_include_params", type=int, default=0)
    ap.add_argument("--param.llm_candidate_pool", dest="param_llm_candidate_pool", type=int, default=30)

    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)

    params: Dict[str, Any] = {
        "base_rate": args.param_base_rate,
        "beta_E": args.param_beta_E,
        "beta_friction": args.param_beta_friction,
        "top_k": args.param_top_k,
        "weight_rule": args.param_weight_rule,
        "rank_weights": args.param_rank_weights,
        "alpha": args.param_alpha,
        "add_if_absent": args.param_add_if_absent,
        "p_new": args.param_p_new,
        "llm_include_params": int(args.param_llm_include_params),
    "llm_candidate_pool": int(args.param_llm_candidate_pool),
    }

    if args.use_best:
        with open(args.use_best, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if "top_k" in loaded:
            try:
                loaded["top_k"] = int(float(loaded["top_k"]))
            except Exception:
                pass
        params.update(loaded)

    llm_seed = args.seed if args.llm_seed is None else args.llm_seed
    llm_cfg = LLMConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        temperature=float(args.llm_temperature),
        top_p=float(args.llm_top_p),
        max_tokens=int(args.llm_max_tokens),
        seed=int(llm_seed),
    )

    edges_out = run_simulate(
        nodes=nodes,
        observed_edges=edges,
        outdir=args.outdir,
        T=int(args.T),
        params=params,
        seed=int(args.seed),
        use_persona=(args.ablate_persona == 0),
        use_climate=(args.ablate_climate == 0),
        llm_cfg=llm_cfg,
        init_graph=str(args.init_graph),
    )

    sim_edges = pd.read_csv(edges_out)
    sim_metrics = compute_basic_metrics(nodes, sim_edges)

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sim_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(sim_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

