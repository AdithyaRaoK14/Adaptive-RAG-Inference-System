"""
adaptive/cache.py
-----------------
LRU (Least Recently Used) cache for exact query matches.
Repeated queries return instantly — 0ms vs 2-4 seconds.

Key is normalised (lowercased, whitespace-collapsed) so
"What is ML?" and "what is ml ?" hit the same cache entry.
"""

from __future__ import annotations
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    answer: str
    chunks: list
    timestamp: float
    hit_count: int = 0


class QueryCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[CacheEntry]:
        key = self._norm(query)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key].hit_count += 1
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, query: str, answer: str, chunks: list) -> None:
        key = self._norm(query)
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)
        self._store[key] = CacheEntry(answer=answer, chunks=chunks, timestamp=time.time())

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    @staticmethod
    def _norm(query: str) -> str:
        return re.sub(r"\s+", " ", query.lower().strip())
