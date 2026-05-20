"""
main.py - Run: python main.py
"""

import os
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate

colorama_init(autoreset=True)

from config import RAGConfig
from pipeline import AdaptiveRAGPipeline

QUERIES = [
    # simple  → llama3.2:3b
    "What is supervised learning?",
    "What is FAISS?",
    "What is ReLU?",
    # moderate → llama3.2:3b
    "How does BM25 differ from vector search?",
    "What are common techniques to prevent overfitting?",
    "How does batch normalisation help neural network training?",
    # complex → qwen2.5:7b
    (
        "Compare dense retrieval and sparse retrieval in RAG systems, "
        "including their strengths, weaknesses, and how hybrid retrieval combines them."
    ),
    (
        "Explain how the Transformer architecture works and why it replaced "
        "RNNs for sequence modelling tasks."
    ),
]

def sep(title=""):
    print(f"\n{Fore.CYAN}{'─'*65}")
    if title:
        print(f"  {title}")
        print(f"{'─'*65}{Style.RESET_ALL}")
    else:
        print(Style.RESET_ALL, end="")

def print_result(result, idx):
    cache_tag = f" {Fore.GREEN}[CACHED]{Style.RESET_ALL}" if result.from_cache else ""
    print(f"\n{Fore.YELLOW}[Q{idx+1}]{Style.RESET_ALL} {result.query[:80]}{cache_tag}")
    print(f"  {Fore.BLUE}Plan   :{Style.RESET_ALL} {result.plan.notes}")
    print(f"  {Fore.BLUE}Type   :{Style.RESET_ALL} {result.analysis.query_type} "
          f"(score={result.analysis.complexity_score:.2f}, words={result.analysis.word_count})")
    print(f"  {Fore.MAGENTA}Model  :{Style.RESET_ALL} {result.model_used}")
    if result.sub_questions:
        print(f"  {Fore.MAGENTA}Decomposed into {len(result.sub_questions)} sub-questions{Style.RESET_ALL}")
    print(f"  {Fore.BLUE}Timing :{Style.RESET_ALL} "
          f"retrieval={result.retrieval_time:.2f}s  "
          f"generation={result.generation_time:.2f}s  "
          f"total={result.total_time:.2f}s")
    print(f"\n  {Fore.WHITE}Answer:{Style.RESET_ALL}")
    for line in result.answer.split("\n"):
        print(f"    {line}")

def print_report(pipeline):
    sep("Performance Report")
    report = pipeline.performance_report()
    if not report:
        print("  No queries recorded.")
        return

    timing = [
        ["P50 Total Latency",   f"{report.get('p50_latency',  0):.3f}s"],
        ["P95 Total Latency",   f"{report.get('p95_latency',  0):.3f}s"],
        ["P50 Retrieval Time",  f"{report.get('p50_ret_time', 0):.3f}s"],
        ["P95 Retrieval Time",  f"{report.get('p95_ret_time', 0):.3f}s"],
        ["P50 Generation Time", f"{report.get('p50_gen_time', 0):.3f}s"],
        ["P95 Generation Time", f"{report.get('p95_gen_time', 0):.3f}s"],
    ]
    print(tabulate(timing, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    adaptive = [
        ["Queries run",    report.get("n_queries",  0)],
        ["Avg quality",    f"{report.get('avg_quality', 0):.3f}"],
        ["Avg top-K used", f"{report.get('avg_top_k',  0):.1f}"],
        ["Avg alpha",      f"{report.get('avg_alpha',  0):.3f}"],
    ]
    print(tabulate(adaptive, headers=["Adaptive Metric", "Value"], tablefmt="rounded_outline"))

    cache = report.get("cache", {})
    if cache:
        cache_rows = [
            ["Cache size",   cache.get("size",     0)],
            ["Cache hits",   cache.get("hits",     0)],
            ["Cache misses", cache.get("misses",   0)],
            ["Hit rate",     f"{cache.get('hit_rate', 0):.1%}"],
        ]
        print(tabulate(cache_rows, headers=["Cache", "Value"], tablefmt="rounded_outline"))

    sep("Model Routing Summary")
    print(f"  simple / moderate queries  →  {Fore.GREEN}llama3.2:3b{Style.RESET_ALL}  (fast)")
    print(f"  complex queries            →  {Fore.YELLOW}qwen2.5:7b{Style.RESET_ALL}   (accurate)")

def main():
    sep("Adaptive RAG System  —  Ollama + Model Routing")

    cfg = RAGConfig(
        small_model="llama3.2:3b",
        large_model="qwen2.5:7b",
        enable_model_routing=True,
        enable_decomposition=True,
        enable_cache=True,
        decomposition_complexity_threshold=0.65,
    )
    pipeline = AdaptiveRAGPipeline(config=cfg)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pipeline.ingest_directory(data_dir)

    sep("Running Queries")
    for i, q in enumerate(QUERIES):
        try:
            result = pipeline.query(q)
            print_result(result, i)
        except Exception as e:
            print(f"{Fore.RED}[Q{i+1}] ERROR: {e}{Style.RESET_ALL}")

    sep("Cache Demo — repeating Q1")
    print_result(pipeline.query(QUERIES[0]), 99)

    print_report(pipeline)

if __name__ == "__main__":
    main()
