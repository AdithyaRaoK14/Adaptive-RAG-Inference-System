[![Tests](https://github.com/AdithyaRaoK14/Adaptive-RAG-Inference-System/actions/workflows/tests.yml/badge.svg)](https://github.com/AdithyaRaoK14/Adaptive-RAG-Inference-System/actions/workflows/tests.yml)


# Adaptive RAG Inference System

A Retrieval-Augmented Generation pipeline that optimises itself at inference time — no training required. Built with FAISS, BM25, sentence-transformers, and local Ollama LLMs.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           User Query                    │
                        └─────────────────┬───────────────────────┘
                                          │
                        ┌─────────────────▼───────────────────────┐
                        │         LRU Query Cache                 │
                        │   (exact match → return instantly)      │
                        └──────┬──────────────────────────────────┘
                               │ MISS
                        ┌──────▼──────────────────────────────────┐
                        │        QueryAnalyzer                    │
                        │  5 signals → complexity score [0,1]     │
                        │  → simple / moderate / complex          │
                        └──────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │        DecisionLayer                    │
                        │  complexity + feedback → top-K, alpha   │
                        │  simple→K=2  moderate→K=5  complex→K=12 │
                        └──────┬──────────────────────────────────┘
                               │
               ┌───────────────▼──────────────────┐
               │         complex query?            │
               │      QueryDecomposer (LLM)        │
               │   splits into 2-4 sub-questions   │
               └───────────────┬──────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │        HybridRetriever                  │
                        │  alpha × FAISS(dense) +                 │
                        │  (1-alpha) × BM25(sparse)               │
                        └──────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │       HeuristicReranker                 │
                        │  coverage + length + position signals   │
                        └──────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │     Generator  (Model Routing)          │
                        │  simple/moderate → llama3.2:3b (fast)   │
                        │  complex        → qwen2.5:7b (accurate) │
                        └──────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │        FeedbackLoop                     │
                        │  EMA tracking → nudge K and alpha       │
                        └──────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────────────────────────────────┐
                        │           Answer + Metadata             │
                        └─────────────────────────────────────────┘
```

---

## File Structure

```
adaptive_rag/
├── config.py                        ← all settings (models, K, alpha, thresholds)
├── pipeline.py                      ← main orchestrator
├── main.py                          ← run this for the full demo
├── requirements.txt
│
├── ingestion/
│   └── document_loader.py           ← text ingestion + sentence-aware chunking
│
├── retrieval/
│   ├── vector_store.py              ← FAISS IndexFlatIP (cosine similarity)
│   ├── keyword_search.py            ← BM25Okapi sparse retrieval
│   ├── hybrid_retriever.py          ← weighted score fusion of dense + sparse
│   ├── reranker.py                  ← heuristic reranker (coverage + position)
│   └── ann_experiments.py           ← ANN benchmark: IVF / HNSW / PQ tuning
│
├── adaptive/
│   ├── query_analyzer.py            ← rule-based complexity scorer (5 signals)
│   ├── decision_layer.py            ← runtime K and alpha selector
│   ├── feedback.py                  ← EMA latency + quality tracking
│   ├── cache.py                     ← LRU query cache
│   └── decomposer.py                ← multi-step query decomposition
│
├── generation/
│   └── generator.py                 ← Ollama wrapper + model routing logic
│
├── data/
│   ├── ml.txt                       ← sample corpus: machine learning
│   ├── rag.txt                      ← sample corpus: RAG systems
│   └── neural_nets.txt              ← sample corpus: neural networks
│
└── tests/
    └── test_all.py                  ← 34 unit tests (no Ollama needed)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Models pulled:

```powershell
ollama pull llama3.2:3b    # ~2GB  — used for simple/moderate queries
ollama pull qwen2.5:7b     # ~5GB  — used for complex queries
```

### Install

```powershell
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

### Run

```powershell
# Full demo (8 queries + performance report)
python main.py

# ANN tuning benchmark
python retrieval/ann_experiments.py

# Unit tests (no Ollama needed)
pytest tests/ -v
```

---

## How It Works

### Part 1 — Basic Pipeline

Documents are loaded from `data/`, split into overlapping 400-character chunks using a sentence-boundary-aware chunker, embedded using `all-MiniLM-L6-v2` (384-dim), and stored in a FAISS `IndexFlatIP`. At query time: embed query → retrieve chunks → generate answer via Ollama.

### Part 2 — Retrieval Optimisation

**Hybrid retrieval** combines two signals:

| Signal | Method | Strength |
|---|---|---|
| Dense | FAISS cosine similarity | Semantic / paraphrase matching |
| Sparse | BM25Okapi | Exact keywords, proper nouns, acronyms |

Combined as: `score = alpha × dense_score + (1-alpha) × bm25_score`

Default `alpha = 0.70`. The DecisionLayer adjusts alpha at runtime — keyword-heavy queries get lower alpha (more BM25), conceptual queries get higher alpha (more FAISS).

**Re-ranking** applies three heuristic signals after retrieval:
- Query term coverage (fraction of query tokens in chunk)
- Length signal (penalise very short / partial chunks)
- Position bonus (earlier chunks in a doc tend to be more definitional)

Blend: `final_score = 0.7 × retrieval_score + 0.3 × heuristic_score`

### Part 3 — Adaptive Decision Layer

The `QueryAnalyzer` scores every query across 5 weighted signals:

| Signal | Weight | What it captures |
|---|---|---|
| Word count | 30% | More words = likely more complex |
| Question depth | 30% | "how/why/compare/explain" starters + WH-word count |
| Conjunction load | 15% | "and/but/however" = multi-aspect query |
| Specificity | 15% | Proper nouns, acronyms, numbers, quoted terms |
| Clause count | 10% | Commas/semicolons as sub-question proxies |

The `DecisionLayer` maps complexity score → K and alpha:

```
score < 0.35  →  simple   →  K = 2   (min)
score 0.35–0.65  →  moderate  →  K = 5   (default)
score > 0.65  →  complex  →  K = 12  (max)

high specificity signal  →  alpha -= 0.15  (lean more on BM25)
high latency detected    →  K reduced by k_bump
low quality detected     →  K increased by k_bump
```

### Part 4 — Feedback Loop

Tracks three metrics using **Exponential Moving Average** (α=0.25):
- `ema_latency` — smoothed total response time
- `ema_quality` — smoothed quality proxy score
- `ema_ret_time` — smoothed retrieval time

**Quality proxy** (no ground truth needed):

| Signal | Weight | Rationale |
|---|---|---|
| Answer length | 40% | Very short = likely "I don't know" |
| Context utilisation | 35% | Do chunk keywords appear in the answer? |
| Confidence words | 25% | Penalise "I'm not sure", reward "according to" |

**Adjustment fires every 3 queries:**
```
ema_latency > 2.5s   →  K -= k_bump   (system too slow, cut retrieval)
ema_quality < 0.35   →  K += k_bump   (poor answers, need more context)
retrieval > 60% time →  K -= 1, alpha -= 0.05
```

### Part 5 — Performance Results

Tested on 8 queries (3 simple, 3 moderate, 2 complex) over a 3-document corpus (26 chunks).

#### Latency

| Metric | Value |
|---|---|
| P50 Total Latency | 6.774s |
| P95 Total Latency | 74.334s |
| P50 Retrieval Time | 0.016s |
| P95 Retrieval Time | 0.040s |
| P50 Generation Time | 6.755s |
| P95 Generation Time | 74.333s |

**Key insight:** Retrieval is only ~0.016s (P50). Generation dominates at ~6.75s (P50). Reducing top-K has minimal impact on total latency — the bottleneck is the LLM, not FAISS.

#### Adaptive Behaviour Observed

| Query | Type | Score | K Used | Model | Time |
|---|---|---|---|---|---|
| "What is supervised learning?" | simple | 0.09 | 2 | llama3.2:3b | 11.6s |
| "What is FAISS?" | simple | 0.12 | 2 | llama3.2:3b | 6.5s |
| "What is ReLU?" | simple | 0.09 | 2 | llama3.2:3b | 5.0s |
| "How does BM25 differ from vector search?" | moderate | 0.38 | 2 | llama3.2:3b | 6.9s |
| "Common techniques to prevent overfitting?" | simple | 0.20 | 2 | llama3.2:3b | 6.6s |
| "How does batch normalisation help?" | moderate | 0.42 | 2 | llama3.2:3b | 5.5s |
| "Compare dense vs sparse retrieval…" | complex | 0.77 | 5 | qwen2.5:7b | 109s |
| "Explain Transformer architecture…" | complex | 0.70 | 5 | qwen2.5:7b | 58s |

**Feedback loop fired:** `k_delta=-2` after Q2 and Q5 as EMA latency exceeded threshold (~9s). This automatically reduced K for subsequent simple queries.

#### Cache Performance

| Metric | Value |
|---|---|
| Cache size | 8 entries |
| Cache hits | 1 |
| Cache misses | 8 |
| Hit rate | 11.1% |
| Cached query latency | 0.00s |

Repeated Q1 returned instantly (0.00s vs 11.6s original) — effectively infinite speedup.

---

## Bonus Features

### Query Decomposition
Complex queries are split into 2–4 atomic sub-questions, each retrieved and answered independently, then synthesised into one final answer.

```
Q7: "Compare dense retrieval and sparse retrieval in RAG systems..."
  → Sub-Q1: "What is dense retrieval in RAG systems?"
  → Sub-Q2: "What is sparse retrieval in RAG systems?"
  → Sub-Q3: "What are the strengths and weaknesses of dense retrieval?"
  → Sub-Q4: "How does hybrid retrieval combine dense and sparse methods?"
  → Synthesise → final answer
```

### Model Routing
Two Ollama models selected per query at runtime:

| Query Type | Model | Avg Speed | Rationale |
|---|---|---|---|
| simple / moderate | llama3.2:3b | ~6s | Fast enough; saves the bigger model for where it matters |
| complex | qwen2.5:7b | ~20-80s | Better reasoning for multi-aspect questions |

### ANN Tuning Experiments
Benchmark across 10,000 vectors, 200 queries, K=5:

| Index Type | Settings | P50 (ms) | P95 (ms) | Recall@5 | Notes |
|---|---|---|---|---|---|
| IndexFlatIP (exact) | — | 0.701 | 0.944 | 100.0% | Ground truth baseline |
| IndexIVFFlat | nprobe=1 | 0.026 | 0.048 | 4.5% | Very fast, poor recall |
| IndexIVFFlat | nprobe=5 | 0.079 | 0.116 | 15.1% | Better recall |
| IndexIVFFlat | nprobe=10 | 0.110 | 0.152 | 25.9% | Balanced |
| IndexIVFFlat | nprobe=20 | 0.234 | 0.345 | 42.1% | Good recall |
| IndexIVFFlat | nprobe=50 | 0.491 | 0.685 | 74.3% | Best IVF recall |
| IndexHNSWFlat | M=32 | 0.152 | 0.219 | 29.6% | Fast queries, 683ms build |
| IndexIVFPQ | M=32, bits=8 | 0.190 | 0.200 | 11.2% | Smallest RAM, low recall |

**Recommendation:**
- ≤100k chunks → `IndexFlatIP` (exact, simple)
- 100k–1M chunks → `IndexHNSWFlat` (best speed/recall tradeoff)
- 1M+ chunks → `IndexIVFPQ` (fits in RAM, accept recall loss)

---

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| Character-based chunking (400 chars) | No tokenizer dependency at ingest time | Not token-exact; may vary slightly per model |
| Sentence-boundary splitting | Preserves context coherence | Chunks can vary in size |
| FAISS IndexFlatIP | Exact search; simple and reliable for small corpora | Doesn't scale past ~100k chunks |
| BM25Okapi | Handles TF saturation + doc length normalisation | No semantic understanding |
| Weighted score fusion | Interpretable; alpha is a tunable knob | Score scale differences between retrievers |
| Heuristic reranker | <1ms overhead; no extra model needed | ~15-20% worse than a cross-encoder |
| EMA feedback (α=0.25) | Smooth adaptation; avoids reacting to single outliers | Slow to react to sudden latency spikes |
| Rule-based query complexity | Zero latency; fully explainable | Less accurate than an LLM-based classifier |
| Quality proxy (no labels) | Works without any ground truth data | Weak signal; noisy on short answers |
| Model routing (3b vs 7b) | Simple queries don't need heavy models | 3b model less accurate on nuanced questions |

---

## What Worked / What Didn't

### ✅ What Worked

- **Hybrid retrieval** consistently outperformed either BM25 or vector search alone on mixed query types
- **EMA feedback** converged quickly (2-3 queries) and correctly reduced K when latency exceeded the threshold
- **Model routing** cut simple query time from ~22s (qwen2.5:7b) to ~6s (llama3.2:3b) — 3.6× speedup
- **Query decomposition** produced noticeably better answers for multi-aspect questions (Q7, Q8)
- **LRU cache** gave 0.00s response on repeated queries — effectively infinite speedup
- **Heuristic reranker** improved top-1 relevance with zero latency overhead

### ❌ What Didn't / Limitations

- **Quality proxy is noisy** — answer length and keyword overlap are weak signals; a small labelled evaluation set + RAGAS would give much stronger feedback
- **EMA is slow to react** to sudden spikes (α=0.25 is conservative); a sliding window would respond faster
- **Generation dominates latency** — reducing K barely helps total latency since retrieval is already ~0.016s; LLM speed is the real bottleneck
- **No index persistence** — FAISS index is rebuilt on every run; production use needs `faiss.write_index()` + pickle
- **Decomposition latency is high** — 2-4 extra LLM calls per complex query; needs a tighter complexity threshold in latency-sensitive deployments
- **IVF recall is low** — nlist=100 on only 10k vectors is too many clusters; rule of thumb is nlist ≈ sqrt(N) which gives better recall at this scale

### How the System Adapts

```
Query arrives
    │
    ├─ Complexity score  →  sets base K  (2 / 5 / 12)
    ├─ Specificity score →  adjusts alpha (more BM25 if keyword-heavy)
    ├─ Model routing     →  picks llama3.2:3b or qwen2.5:7b
    │
    ▼ After response:
    ├─ EMA latency updates every query
    ├─ EMA quality updates every query
    │
    └─ Every 3 queries: adjustment fires
           ema_latency > 2.5s  →  K -= 2  (seen in demo: fired after Q2 and Q5)
           ema_quality < 0.35  →  K += 2
```

---

## Test Results

```
34 passed, 3 warnings in 0.37s
```

All 34 unit tests pass with no API key and no Ollama connection required. Tests cover: DocumentLoader, KeywordSearch, VectorStore, HybridRetriever, HeuristicReranker, QueryAnalyzer, DecisionLayer, and QueryCache.

---

## Dependencies

| Package | Purpose |
|---|---|
| faiss-cpu | Vector similarity search |
| sentence-transformers | Embedding model (all-MiniLM-L6-v2) |
| rank-bm25 | BM25 sparse retrieval |
| requests | Ollama API calls |
| numpy | Vector operations |
| tabulate | Performance report tables |
| colorama | Coloured terminal output |
| pytest | Unit testing |
