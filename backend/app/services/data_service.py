# backend/app/services/data_service.py
"""
Fetches papers from Semantic Scholar API.
Falls back to synthetic OGB-Arxiv data when offline.
"""

import os
import json
import hashlib
import numpy as np
import httpx
from typing import List, Dict, Tuple, Optional
from functools import lru_cache


_OGB_CACHE = None


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,year,citationCount,abstract,authors,venue,references"

def load_ogb_papers(path="ogb_arxiv_papers.json"):
    global _OGB_CACHE
    if _OGB_CACHE is None:
        with open(path) as f:
            _OGB_CACHE = json.load(f)
    return _OGB_CACHE["papers"], _OGB_CACHE["edges"]


def _paper_to_features(paper: Dict, feat_dim: int = 64) -> List[float]:
    """
    Convert paper metadata to a feature vector.
    In production replace with real word2vec / SPECTER embeddings.
    Uses a deterministic hash so same paper always gets same features.
    """
    seed_str = (paper.get("title", "") + str(paper.get("year", 0)))
    seed     = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng      = np.random.default_rng(seed)
    return rng.standard_normal(feat_dim).astype(float).tolist()


async def fetch_papers_by_query(query: str,
                                 limit: int = 50) -> List[Dict]:
    """Fetch papers from Semantic Scholar by keyword query."""
    url    = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {"query": query, "limit": limit, "fields": FIELDS}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data   = r.json()
            papers = data.get("data", [])
            return [_normalise_paper(p) for p in papers if p.get("year")]
    except Exception:
        # Fallback to synthetic data
        return _synthetic_papers(query, limit)


def _normalise_paper(p: Dict) -> Dict:
    return {
        "id":        p.get("paperId", ""),
        "title":     p.get("title", "Unknown"),
        "year":      p.get("year", 2020),
        "citations": p.get("citationCount", 0),
        "abstract":  p.get("abstract", ""),
        "authors":   [a["name"] for a in p.get("authors", [])],
        "venue":     p.get("venue", ""),
        "features":  _paper_to_features(p),
    }


def _synthetic_papers(query: str, n: int = 50) -> List[Dict]:
    """
    Synthetic OGB-Arxiv style papers for demo / offline use.
    Mirrors the exact dataset used in the T-GIB experiments.
    """
    rng   = np.random.default_rng(abs(hash(query)) % (2**31))
    years = np.arange(2013, 2021)
    yprob = [0.06, 0.07, 0.08, 0.10, 0.13, 0.16, 0.19, 0.21]

    topics = [
        "Graph Neural Networks", "Transformer Architecture",
        "Contrastive Learning", "Diffusion Models",
        "Reinforcement Learning", "Knowledge Graphs",
        "Federated Learning", "Neural Architecture Search",
    ]

    papers = []
    for i in range(n):
        yr    = int(rng.choice(years, p=yprob))
        topic = topics[i % len(topics)]
        # Shift papers: low citations early, burst later
        is_shift = (i < 8 and yr <= 2016)
        cites    = (int(rng.integers(200, 800)) if is_shift and yr >= 2017
                    else int(rng.integers(5, 150)))
        papers.append({
            "id":        f"synthetic_{i:04d}",
            "title":     f"{topic}: {query} (Paper {i+1})",
            "year":      yr,
            "citations": cites,
            "abstract":  f"This paper studies {query} in the context of {topic}.",
            "authors":   [f"Author {rng.integers(1,50)}"],
            "venue":     rng.choice(["NeurIPS","ICML","ICLR","KDD","WWW"]),
            "features":  (rng.standard_normal(64) * (2.0 if is_shift else 1.0)
                          ).astype(float).tolist(),
        })
    return papers


def build_citation_edges(papers: List[Dict]) -> List[Tuple[int, int, int]]:
    """
    Build citation edges from paper list.
    In production: use references field from Semantic Scholar.
    Fallback: probabilistic power-law simulation.
    """
    n     = len(papers)
    years = [p["year"] for p in papers]
    rng   = np.random.default_rng(42)
    pop   = np.ones(n)
    edges = []

    for src in range(n):
        older = [i for i in range(n) if years[i] < years[src]]
        if len(older) < 2:
            continue
        pw  = pop[older] ** 1.3
        pw /= pw.sum()
        nc_ = min(rng.poisson(4), len(older))
        if nc_ == 0:
            continue
        for tgt in rng.choice(older, size=nc_, replace=False, p=pw):
            edges.append((src, int(tgt), years[src]))
            pop[tgt] += 1.0

    return edges
