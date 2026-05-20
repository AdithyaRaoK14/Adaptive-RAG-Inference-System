"""
generation/generator.py
-----------------------
Calls local Ollama to generate answers.

MODEL ROUTING (bonus):
  simple / moderate query  →  small_model  (llama3.2:3b  — fast)
  complex query            →  large_model  (qwen2.5:7b   — accurate)

The pipeline passes query_type at call time so routing is per-query.
"""

from __future__ import annotations
import time
import requests
from typing import List, Tuple, Optional
from config import RAGConfig
from ingestion.document_loader import Chunk

SYSTEM_PROMPT = """You are a precise, factual question-answering assistant.
You are given retrieved document passages as context.
Answer the user's question using ONLY information from these passages.
If the passages don't contain enough information, say so explicitly.
Cite sources using [Source: <name>] when relevant.
Be concise but complete. Do not speculate beyond the provided context."""


class Generator:
    def __init__(self, config: RAGConfig):
        self.cfg = config
        self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.cfg.ollama_base_url}/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            available_base = [m.split(":")[0] for m in available]
            for model in [self.cfg.small_model, self.cfg.large_model]:
                if model.split(":")[0] not in available_base:
                    print(f"[WARN] Model '{model}' not found. Run: ollama pull {model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "\n[ERROR] Cannot connect to Ollama at http://localhost:11434\n"
                "        Open the Ollama app first.\n"
            )

    def route_model(self, query_type: str) -> str:
        """
        Model routing logic:
          simple / moderate → small_model (llama3.2:3b) — faster
          complex           → large_model (qwen2.5:7b)  — more accurate
        """
        if not self.cfg.enable_model_routing:
            return self.cfg.large_model
        if query_type == "complex":
            return self.cfg.large_model
        return self.cfg.small_model

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        query_type: str = "moderate",
    ) -> Tuple[str, float, str]:
        """
        Returns (answer_text, elapsed_seconds, model_used).
        query_type drives model routing.
        """
        model = self.route_model(query_type)
        context = self._build_context(retrieved)
        user_msg = f"Context passages:\n{context}\n\nQuestion: {query}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.cfg.ollama_base_url}/api/chat",
            json=payload,
            timeout=180,
        )
        r.raise_for_status()
        elapsed = time.perf_counter() - t0

        answer = r.json().get("message", {}).get("content", "")
        return answer, elapsed, model

    @staticmethod
    def _build_context(retrieved: List[Tuple[Chunk, float]]) -> str:
        parts = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            header = f"[{rank}] Source: {chunk.source} | score={score:.3f}"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)
