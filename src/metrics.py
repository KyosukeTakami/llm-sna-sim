import pandas as pd, numpy as np
import networkx as nx
from typing import Dict
from scipy.stats import ks_2samp, pearsonr

def graph_from(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r['student_id'], **r.to_dict())
    for _, r in edges.iterrows():
        G.add_edge(r['source'], r['target'], weight=float(r.get('weight', 1.0)))
    return G

def compute_basic_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> Dict:
    G = graph_from(nodes, edges)
    deg = dict(G.degree())
    nodes = nodes.copy()
    nodes["deg"] = nodes["student_id"].map(deg).fillna(0)
    deg_sim = nodes["deg"].to_numpy()

    density = nx.density(G)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.quality.modularity(G, comms) if len(comms) > 1 else 0.0
        n_comms = len(comms)
    except Exception:
        modularity, n_comms = float("nan"), 0

    iso_rate = float((nodes["deg"] == 0).mean())

    reciprocity = 0.0
    if G.number_of_edges() > 0:
        reciprocity = float(np.mean([
            1.0 if float(d.get("weight", 1.0)) >= 2.0 else 0.0
            for _, _, d in G.edges(data=True)
        ]))

    # ---- SAFE pearsonr ----
    x = nodes["BFI_E_z"].fillna(0.0).to_numpy(dtype=float)
    y = nodes["deg"].fillna(0.0).to_numpy(dtype=float)

    r = 0.0  # フォールバック値（NaNにしない）
    if len(x) >= 2 and len(y) >= 2:
        # 分散ゼロ（定数列）だと pearsonr が定義できないので弾く
        if (np.nanstd(x) > 1e-12) and (np.nanstd(y) > 1e-12):
            try:
                r_tmp, _ = pearsonr(x, y)
                if np.isfinite(r_tmp):
                    r = float(r_tmp)
            except Exception:
                r = 0.0

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": float(density),
        "modularity": float(modularity),
        "n_communities": int(n_comms),
        "isolation_rate": float(iso_rate),
        "reciprocity_rate": float(reciprocity),
        "r_extraversion_degree": float(r),
        "degree_list": deg_sim.tolist(),
    }



def sfi_distance(sim: Dict, obs: Dict, weights: Dict) -> float:
    ks = 0.0
    if len(sim.get("degree_list",[]))>0 and len(obs.get("degree_list",[]))>0:
        ks = ks_2samp(sim["degree_list"], obs["degree_list"]).statistic
    dens = abs(sim["density"] - obs["density"])
    mod = abs(sim["modularity"] - obs["modularity"])
    rdiff = abs(sim["r_extraversion_degree"] - obs["r_extraversion_degree"])
    iso = abs(sim["isolation_rate"] - obs["isolation_rate"])
    rec = abs(sim["reciprocity_rate"] - obs["reciprocity_rate"])
    w = weights
    return (w["ks_degree"]*ks + w["density"]*dens + w["modularity"]*mod +
            w["r_extraversion_degree"]*rdiff + w["isolation"]*iso + w["reciprocity"]*rec)
