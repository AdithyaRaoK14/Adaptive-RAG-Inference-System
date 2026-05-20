"""
pipeline.py
-----------
Main orchestrator. Wires every component together.

Query flow:
  1. Cache check          → return instantly if seen before
  2. QueryAnalyzer        → score complexity (simple/moderate/complex)
  3. DecisionLayer        → pick top-K and alpha
  4. QueryDecomposer      → split if very complex (bonus)
  5. HybridRetriever      → dense (FAISS) + sparse (BM25)
  6. HeuristicReranker    → reorder results
  7. Generator            → call Ollama with MODEL ROUTING
                            simple/moderate → llama3.2:3b
                            complex         → qwen2.5:7b
  8. FeedbackLoop         → update EMA, nudge K and alpha
  9. Return RAGResult
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from ingestion.document_loader import DocumentLoader, Chunk
from retrieval.vector_store import VectorStore
from retrieval.keyword_search import KeywordSearch
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import HeuristicReranker
from adaptive.query_analyzer import QueryAnalyzer, QueryAnalysis
from adaptive.decision_layer import DecisionLayer, RetrievalPlan
from adaptive.feedback import FeedbackLoop
from adaptive.cache import QueryCache
from adaptive.decomposer import QueryDecomposer
from generation.generator import Generator


@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: List[Tuple[Chunk, float]]
    plan: RetrievalPlan
    analysis: QueryAnalysis
    retrieval_time: float
    generation_time: float
    total_time: float
    model_used: str = ""
    from_cache: bool = False
    sub_questions: List[str] = field(default_factory=list)


class AdaptiveRAGPipeline:
    def __init__(self, config: RAGConfig = None):
        self.cfg = config or RAGConfig()

        print("[Pipeline] Loading embedding model…")
        self.embedder         = SentenceTransformer(self.cfg.embedding_model)
        self.vector_store     = VectorStore(self.cfg.embedding_dim)
        self.keyword_search   = KeywordSearch()
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.keyword_search, self.cfg.default_alpha
        )
        self.reranker         = HeuristicReranker()
        self.query_analyzer   = QueryAnalyzer(self.cfg)
        self.decision_layer   = DecisionLayer(self.cfg)
        self.feedback_loop    = FeedbackLoop(self.cfg, self.decision_layer)
        self.cache            = QueryCache(self.cfg.cache_max_size)
        self.generator        = Generator(self.cfg)

        self.decomposer = None
        if self.cfg.enable_decomposition:
            self.decomposer = QueryDecomposer(
                model=self.cfg.large_model,
                base_url=self.cfg.ollama_base_url,
            )
        self._built = False

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, texts: List[str], sources: List[str] = None) -> None:
        chunks = DocumentLoader(self.cfg).load_texts(texts, sources)
        self._index(chunks)

    def ingest_directory(self, dir_path: str) -> None:
        chunks = DocumentLoader(self.cfg).load_directory(dir_path)
        self._index(chunks)

    def _index(self, chunks: List[Chunk]) -> None:
        print(f"[Pipeline] Embedding {len(chunks)} chunks…")
        embs = self.embedder.encode(
            [c.text for c in chunks],
            show_progress_bar=True, batch_size=64, normalize_embeddings=True
        )
        self.vector_store.build(chunks, embs)
        self.keyword_search.build(chunks)
        self._built = True
        print(f"[Pipeline] Ready — {len(chunks)} chunks indexed.\n")

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResult:
        if not self._built:
            raise RuntimeError("Call ingest() first.")

        t_start = time.perf_counter()

        # 1. Cache
        if self.cfg.enable_cache:
            cached = self.cache.get(question)
            if cached:
                dummy_plan     = RetrievalPlan(self.cfg.default_top_k, self.cfg.default_alpha, "cache_hit")
                dummy_analysis = self.query_analyzer.analyse(question)
                print(f"[Cache] HIT")
                return RAGResult(
                    query=question, answer=cached.answer,
                    retrieved_chunks=[(c, 0.0) for c in cached.chunks],
                    plan=dummy_plan, analysis=dummy_analysis,
                    retrieval_time=0.0, generation_time=0.0,
                    total_time=time.perf_counter() - t_start,
                    model_used="(cached)", from_cache=True,
                )

        # 2. Analyse + Plan
        analysis = self.query_analyzer.analyse(question)
        plan     = self.decision_layer.plan(analysis, self.feedback_loop.stats)
        print(f"[Plan] {plan.notes}")

        # 3. Model routing decision
        model_used = self.generator.route_model(analysis.query_type)
        print(f"[Model] Routing to → {model_used}  (query_type={analysis.query_type})")

        # 4. Decompose if complex
        if (
            self.decomposer
            and analysis.complexity_score >= self.cfg.decomposition_complexity_threshold
        ):
            sub_qs = self.decomposer.decompose(question)
            if len(sub_qs) > 1:
                return self._decomposed(question, sub_qs, plan, analysis, t_start, model_used)

        # 5. Retrieve
        t_ret  = time.perf_counter()
        q_vec  = self.embedder.encode([question], normalize_embeddings=True)[0]
        raw    = self.hybrid_retriever.search(question, q_vec, plan.top_k, plan.alpha)
        ranked = self.reranker.rerank(question, raw, plan.top_k)
        retrieval_time = time.perf_counter() - t_ret

        # 6. Generate (with routed model)
        t_gen = time.perf_counter()
        answer, _, model_used = self.generator.generate(question, ranked, analysis.query_type)
        generation_time = time.perf_counter() - t_gen

        total_time = time.perf_counter() - t_start

        # 7. Cache + Feedback
        if self.cfg.enable_cache:
            self.cache.put(question, answer, [c for c, _ in ranked])
        self.feedback_loop.record(
            query=question, top_k=plan.top_k, alpha=plan.alpha,
            retrieval_time=retrieval_time, generation_time=generation_time,
            answer=answer, retrieved_chunks=[c for c, _ in ranked],
        )

        return RAGResult(
            query=question, answer=answer, retrieved_chunks=ranked,
            plan=plan, analysis=analysis,
            retrieval_time=retrieval_time, generation_time=generation_time,
            total_time=total_time, model_used=model_used,
        )

    def _decomposed(self, main_q, sub_qs, plan, analysis, t_start, model_used) -> RAGResult:
        print(f"[Decomposer] Split into {len(sub_qs)} sub-questions")
        sub_qa, all_chunks = [], []
        t_ret = t_gen = 0.0

        for sq in sub_qs:
            sq_vec = self.embedder.encode([sq], normalize_embeddings=True)[0]
            t0     = time.perf_counter()
            raw    = self.hybrid_retriever.search(sq, sq_vec, plan.top_k, plan.alpha)
            ranked = self.reranker.rerank(sq, raw, plan.top_k)
            t_ret += time.perf_counter() - t0

            t0 = time.perf_counter()
            ans, _, _ = self.generator.generate(sq, ranked, analysis.query_type)
            t_gen += time.perf_counter() - t0

            sub_qa.append((sq, ans))
            all_chunks.extend(ranked)

        t0     = time.perf_counter()
        final  = self.decomposer.synthesise(main_q, sub_qa)
        t_gen += time.perf_counter() - t0

        total = time.perf_counter() - t_start
        if self.cfg.enable_cache:
            self.cache.put(main_q, final, [c for c, _ in all_chunks])
        self.feedback_loop.record(
            query=main_q, top_k=plan.top_k, alpha=plan.alpha,
            retrieval_time=t_ret, generation_time=t_gen,
            answer=final, retrieved_chunks=[c for c, _ in all_chunks],
        )
        return RAGResult(
            query=main_q, answer=final, retrieved_chunks=all_chunks,
            plan=plan, analysis=analysis,
            retrieval_time=t_ret, generation_time=t_gen,
            total_time=total, model_used=model_used,
            sub_questions=sub_qs,
        )

    def performance_report(self) -> dict:
        report = self.feedback_loop.summary()
        report["cache"] = self.cache.stats()
        return report
