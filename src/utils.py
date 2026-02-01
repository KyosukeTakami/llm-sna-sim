import os, json, random, numpy as np, pandas as pd
from typing import Any, List

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def df_required_columns(df: pd.DataFrame, cols: List[str], name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
