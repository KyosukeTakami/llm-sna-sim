#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _norm_pair(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def compute_step_series(log_df: pd.DataFrame, T: int) -> Dict[str, np.ndarray]:
    df = log_df.copy()

    for col in ["t", "source", "target", "delta_w"]:
        if col not in df.columns:
            raise ValueError(f"interaction_log.csv missing column: {col}")

    if "skipped_new_edge" in df.columns:
        df = df[df["skipped_new_edge"] == False]  # noqa: E712

    df = df[df["delta_w"].astype(float) > 0.0]
    df["t"] = df["t"].astype(int)

    existing_pairs = set()

    new_ratio = np.full(T, np.nan, dtype=float)
    recip_event_rate = np.full(T, np.nan, dtype=float)
    recip_dyad_rate = np.full(T, np.nan, dtype=float)

    for t in range(1, T + 1):
        dft = df[df["t"] == t]
        if len(dft) == 0:
            continue

        # --- new-edge ratio ---
        n_new, n_total = 0, 0
        for _, r in dft.iterrows():
            u = str(r["source"])
            v = str(r["target"])
            p = _norm_pair(u, v)
            n_total += 1
            if p not in existing_pairs:
                n_new += 1
            existing_pairs.add(p)

        new_ratio[t - 1] = (n_new / n_total) if n_total > 0 else np.nan

        # --- reciprocity (event-based) ---
        directed = set((str(r["source"]), str(r["target"])) for _, r in dft.iterrows())
        if len(directed) > 0:
            recip_events = sum(1 for (u, v) in directed if (v, u) in directed)
            recip_event_rate[t - 1] = recip_events / len(directed)

        # --- reciprocity (dyad-based) ---
        dyads = {}
        for (u, v) in directed:
            p = _norm_pair(u, v)
            dyads.setdefault(p, set()).add((u, v))

        if len(dyads) > 0:
            recip_dyads = 0
            for (a, b), dirs in dyads.items():
                if (a, b) in dirs and (b, a) in dirs:
                    recip_dyads += 1
            recip_dyad_rate[t - 1] = recip_dyads / len(dyads)

    return {
        "new_ratio": new_ratio,
        "recip_event_rate": recip_event_rate,
        "recip_dyad_rate": recip_dyad_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_rule", required=True)
    ap.add_argument("--tag_llm", required=True)
    ap.add_argument("--classes", default="A,B,C,D")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--runs_root", default="results/runs")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--title", default=None)

    args = ap.parse_args()
    _ensure_dir(args.outdir)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    T = int(args.T)
    x = np.arange(1, T + 1)

    series_rule = {}
    series_llm = {}

    for cls in classes:
        rule_log = os.path.join(args.runs_root, f"{cls}_{args.tag_rule}_best", "interaction_log.csv")
        llm_log  = os.path.join(args.runs_root, f"{cls}_{args.tag_llm}_best", "interaction_log.csv")
        if not os.path.exists(rule_log):
            raise FileNotFoundError(f"Missing: {rule_log}")
        if not os.path.exists(llm_log):
            raise FileNotFoundError(f"Missing: {llm_log}")

        series_rule[cls] = compute_step_series(pd.read_csv(rule_log), T=T)
        series_llm[cls]  = compute_step_series(pd.read_csv(llm_log),  T=T)

    # ---- styling rule ----
    # color = class, linestyle = model
    color_map = {cls: f"C{i}" for i, cls in enumerate(classes)}  # default matplotlib cycle
    ls_rule = "-"
    ls_llm = "--"
    mk_rule = "o"
    mk_llm = "o"

    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    for cls in classes:
        c = color_map[cls]
        ax1.plot(x, series_rule[cls]["new_ratio"],        marker=mk_rule, linestyle=ls_rule, color=c)
        ax1.plot(x, series_llm[cls]["new_ratio"],         marker=mk_llm,  linestyle=ls_llm,  color=c)

        ax2.plot(x, series_rule[cls]["recip_event_rate"], marker=mk_rule, linestyle=ls_rule, color=c)
        ax2.plot(x, series_llm[cls]["recip_event_rate"],  marker=mk_llm,  linestyle=ls_llm,  color=c)

        ax3.plot(x, series_rule[cls]["recip_dyad_rate"],  marker=mk_rule, linestyle=ls_rule, color=c)
        ax3.plot(x, series_llm[cls]["recip_dyad_rate"],   marker=mk_llm,  linestyle=ls_llm,  color=c)

    ax1.set_ylabel("New-edge ratio per step"); ax1.set_xlabel("Step"); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Reciprocity (event-based)"); ax2.set_xlabel("Step"); ax2.grid(True, alpha=0.3)
    ax3.set_ylabel("Reciprocity (dyad-based)");  ax3.set_xlabel("Step"); ax3.grid(True, alpha=0.3)

    # ---- two legends (separate) ----
    class_handles = [
        Line2D([0], [0], color=color_map[cls], lw=2, label=f"Class {cls}") for cls in classes
    ]
    model_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle=ls_rule, label="Rule-based"),
        Line2D([0], [0], color="black", lw=2, linestyle=ls_llm,  label="LLM-based"),
    ]

    # place legends so they never collide with title
    leg1 = fig.legend(handles=class_handles, loc="upper center", ncol=len(classes), frameon=False,
                      bbox_to_anchor=(0.5, 0.965))
    leg2 = fig.legend(handles=model_handles, loc="upper center", ncol=2, frameon=False,
                      bbox_to_anchor=(0.5, 0.935))

    for leg in [leg1, leg2]:
        leg._legend_box.align = "left"

    if args.title:
        fig.suptitle(args.title, y=0.995)
    else:
        fig.suptitle(
            f"Interaction dynamics across classes (Rule vs LLM)\n"
            f"TAG_RULE={args.tag_rule}, TAG_LLM={args.tag_llm}",
            y=0.995
        )

    # leave room on top for title + 2 legends
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out = args.outfile or f"abcd_interaction_dynamics__{args.tag_rule}__vs__{args.tag_llm}.png"
    outpath = os.path.join(args.outdir, out)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("[OK] Saved:", outpath)


if __name__ == "__main__":
    main()