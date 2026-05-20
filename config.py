"""
config.py
---------
All tuneable parameters in one place.
"""

from dataclasses import dataclass


@dataclass
class RAGConfig:
    # --- Embedding model ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Chunking ---
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- Retrieval ---
    default_top_k: int = 5
    min_top_k: int = 2
    max_top_k: int = 12
    default_alpha: float = 0.7

    # --- Adaptive thresholds ---
    short_query_words: int = 4
    complex_query_words: int = 12
    high_latency_threshold: float = 2.5
    low_quality_threshold: float = 0.35

    # --- Feedback ---
    ema_alpha: float = 0.25
    k_bump: int = 2
    alpha_bump: float = 0.1

    # --- Cache ---
    enable_cache: bool = True
    cache_max_size: int = 256

    # --- Ollama model routing (BONUS) ---
    # simple/moderate queries -> small_model (fast, llama3.2:3b)
    # complex queries         -> large_model (accurate, qwen2.5:7b)
    small_model: str = "llama3.2:3b"
    large_model: str = "qwen2.5:7b"
    enable_model_routing: bool = True
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = 1024
    temperature: float = 0.3

    # --- Query decomposition ---
    enable_decomposition: bool = True
    decomposition_complexity_threshold: float = 0.65
