# src/llm_agent.py
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class LLMConfig:
    provider: str = "dummy"          # dummy|openai
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: Optional[int] = None


class BaseLLM:
    def score_candidates(self, agent_ctx: Dict[str, Any], candidates: List[str], top_k: int) -> Dict[str, float]:
        raise NotImplementedError


# =========================
# Dummy LLM (baseline / fallback)
# =========================
class DummyLLM(BaseLLM):
    """
    community を使わない擬似LLM：
    個人特性（Extraversion）・クラス気候（friction）に基づいてスコアを返す
    """

    def score_candidates(self, agent_ctx: Dict[str, Any], candidates: List[str], top_k: int) -> Dict[str, float]:
        E = float(agent_ctx.get("BFI_E_z", 0.0))
        base = float(agent_ctx.get("base_rate", 0.15))

        beta_E = float(agent_ctx.get("beta_E", 0.15))  # ★探索/固定する係数
        beta_fric = float(agent_ctx.get("beta_friction", 0.2))

        fric = float(agent_ctx.get("CCI_friction_z", 0.0)) if agent_ctx.get("use_climate", True) else 0.0

        # 個人の「相互作用しやすさ」(0..1) を作る（communityボーナス無し）
        base_prop = base
        if agent_ctx.get("use_persona", True):
            base_prop += beta_E * E
        base_prop -= beta_fric * max(0.0, fric)

        base_prop = max(0.0, min(1.0, float(base_prop)))

        # 再現性
        rng_seed = agent_ctx.get("rng_seed", None)
        rnd = random.Random(int(rng_seed)) if rng_seed is not None else random

        out: Dict[str, float] = {}
        for cid in candidates:
            # 候補ごとの差は微小ノイズのみ（community情報は入れない）
            s = base_prop + 0.02 * rnd.random()
            out[cid] = max(0.0, min(1.0, float(s)))

        return dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:top_k])


# =========================
# OpenAI LLM
# =========================
# 
class OpenAILLM(BaseLLM):
    """
    OpenAI Responses API を用いた実装（community無し版）。
    - JSONのみ出力
    - パース失敗時は Dummy にフォールバック
    """

    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cfg = cfg

    def _sample_candidates(self, candidates: List[str], max_pool: int, rng_seed: Optional[int]) -> List[str]:
        if len(candidates) <= max_pool:
            return candidates
        if rng_seed is not None:
            rnd = random.Random(int(rng_seed))
            return rnd.sample(candidates, k=max_pool)
        return random.sample(candidates, k=max_pool)

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are a decision-making agent in a classroom social network simulation.\n"
            "Given the input JSON, score each candidate in [0,1] where higher means more likely to interact next.\n"
            "Return ONLY valid JSON. Do not include any extra text.\n\n"
            "Required output JSON schema:\n"
            "{\n"
            '  "scores": [\n'
            '    {"id": "CANDIDATE_ID", "score": 0.0}\n'
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Use only candidate ids provided in input JSON.\n"
            "- Provide up to top_k items, sorted by descending score.\n"
            "- score must be a number between 0 and 1.\n\n"
            "Input JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
        )

    def score_candidates(self, agent_ctx: Dict[str, Any], candidates: List[str], top_k: int) -> Dict[str, float]:
        # ---- candidate pool ----
        max_pool = int(agent_ctx.get("llm_candidate_pool", 30))
        rng_seed = agent_ctx.get("rng_seed", self.cfg.seed)
        cand = self._sample_candidates(candidates, max_pool, rng_seed)

        include_params = bool(agent_ctx.get("llm_include_params", False))

        # ---- base payload（paramsなしがデフォルト）----
        payload = {
            "self_id": agent_ctx.get("student_id"),
            # Big Five
            "BFI_E_z": agent_ctx.get("BFI_E_z"),
            "BFI_A_z": agent_ctx.get("BFI_A_z"),
            "BFI_C_z": agent_ctx.get("BFI_C_z"),
            "BFI_N_z": agent_ctx.get("BFI_N_z"),
            "BFI_O_z": agent_ctx.get("BFI_O_z"),
            # Classroom Climate
            "CCI_friction_z": agent_ctx.get("CCI_friction_z"),
            "CCI_satisfaction_z": agent_ctx.get("CCI_satisfaction_z"),
            "CCI_closeness_z": agent_ctx.get("CCI_closeness_z"),
            "use_persona": bool(agent_ctx.get("use_persona", True)),
            "use_climate": bool(agent_ctx.get("use_climate", True)),
            "candidates": cand,
            "top_k": int(top_k),
        }

        # ★ ここが重要：paramsは「後から条件付きで」追加
        if include_params:
            payload["params"] = {
                "base_rate": agent_ctx.get("base_rate"),
                "beta_E": agent_ctx.get("beta_E"),
                "beta_friction": agent_ctx.get("beta_friction"),
            }

        prompt = self._build_prompt(payload)

        try:
            resp = self.client.responses.create(
                model=self.cfg.model,
                input=prompt,
                temperature=float(self.cfg.temperature),
            )

            text = getattr(resp, "output_text", None)
            if not text:
                raise RuntimeError("No output_text from OpenAI")

            data = json.loads(text)
            pairs = data.get("scores", [])

            out: Dict[str, float] = {}
            for item in pairs:
                cid = item.get("id")
                sc = item.get("score")
                if cid in cand:
                    out[str(cid)] = max(0.0, min(1.0, float(sc)))

            # 足りない分は0埋め
            while len(out) < min(top_k, len(cand)):
                for cid in cand:
                    if cid not in out:
                        out[cid] = 0.0
                        break

            return dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:top_k])

        except Exception:
            agent_ctx["llm_fallback"] = True
            return DummyLLM().score_candidates(agent_ctx, cand, top_k)

def build_llm(cfg: LLMConfig) -> BaseLLM:
    if cfg.provider == "openai":
        return OpenAILLM(cfg)
    return DummyLLM()

