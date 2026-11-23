"""Retrievers implemented from scratch: BM25, char n-gram TF-IDF cosine, and RRF fusion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

STOPWORDS = frozenset(
    ["a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "long", "many", "much", "that", "this", "these", "those", "it", "its", "as", "under", "after", "within", "than", "then", "there", "their", "they", "them", "he", "she", "his", "her", "we", "our", "you", "your", "i", "me", "my", "can", "could", "would", "should", "will", "must", "may", "into", "about", "each"]  # noqa: E501
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed. No stemming."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str
    type: str = "doc"
    status: str = "current"

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


def load_corpus(path: str | Path) -> list[Doc]:
    with open(path, encoding="utf-8") as fh:
        return [Doc(**json.loads(line)) for line in fh if line.strip()]


Ranking = list[tuple[int, float]]  # (doc index, score), best first


class BM25:
    """Okapi BM25 with Lucene-style idf: ln(1 + (N - df + 0.5) / (df + 0.5))."""

    name = "bm25"

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.doc_len = np.asarray(lengths, dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.idf = {
            t: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs)
        for term in dict.fromkeys(tokenize(query)):  # unique query terms, in order
            for i, tf in self.postings.get(term, ()):
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def rank(self, query: str) -> Ranking:
        return _rank_positive(self.scores(query))


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter:
    """Character n-grams of each token padded with spaces (like sklearn's char_wb analyzer)."""
    grams: Counter = Counter()
    for tok in tokenize(text):
        padded = f" {tok} "
        for n in range(n_min, n_max + 1):
            for i in range(len(padded) - n + 1):
                grams[padded[i : i + n]] += 1
    return grams


class CharNgramDense:
    """TF-IDF over character n-grams, L2-normalised dense vectors, cosine similarity."""

    name = "char_ngram_dense"

    def __init__(self, texts: Sequence[str], n_min: int = 3, n_max: int = 5):
        self.n_min, self.n_max = n_min, n_max
        counts = [char_ngrams(t, n_min, n_max) for t in texts]
        self.vocab = {g: j for j, g in enumerate(sorted({g for c in counts for g in c}))}
        df = np.zeros(len(self.vocab))
        for c in counts:
            for g in c:
                df[self.vocab[g]] += 1
        # Smoothed idf, as in sklearn: ln((1 + N) / (1 + df)) + 1
        self.idf = np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0
        self.matrix = np.vstack([self._vector(c) for c in counts]) if texts else np.zeros((0, 0))

    def _vector(self, counts: Counter) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for g, tf in counts.items():
            j = self.vocab.get(g)
            if j is not None:
                vec[j] = tf * self.idf[j]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        return self._vector(char_ngrams(text, self.n_min, self.n_max))

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embed(query)

    def rank(self, query: str) -> Ranking:
        return _rank_positive(self.scores(query))


def rrf_fuse(rankings: Sequence[Sequence], k: int = 60) -> list[tuple]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1.

    Ties keep first-seen order (stable sort), so the output is deterministic.
    """
    fused: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class HybridRRF:
    name = "hybrid_rrf"

    def __init__(self, retrievers: Sequence, k: int = 60):
        self.retrievers, self.k = list(retrievers), k

    def rank(self, query: str) -> Ranking:
        rankings = [[i for i, _ in r.rank(query)] for r in self.retrievers]
        return rrf_fuse(rankings, k=self.k)


def _rank_positive(scores: np.ndarray) -> Ranking:
    order = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


RETRIEVERS = ("bm25", "char_ngram_dense", "hybrid_rrf")


class SearchIndex:
    """All three retrievers over one corpus, with memoised rankings (retrieval is deterministic)."""

    def __init__(self, docs: Sequence[Doc]):
        self.docs = list(docs)
        texts = [d.full_text for d in self.docs]
        bm25, dense = BM25(texts), CharNgramDense(texts)
        self._retrievers = {"bm25": bm25, "char_ngram_dense": dense, "hybrid_rrf": HybridRRF([bm25, dense])}
        self._rank = lru_cache(maxsize=None)(self._rank_uncached)

    def _rank_uncached(self, retriever: str, query: str) -> tuple:
        return tuple(self._retrievers[retriever].rank(query))

    def search(self, retriever: str, query: str, top_k: int) -> list[tuple[Doc, float]]:
        return [(self.docs[i], s) for i, s in self._rank(retriever, query)[:top_k]]
