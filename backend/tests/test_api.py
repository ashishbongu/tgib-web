# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _sample_payload(n=5):
    import numpy as np
    rng = np.random.default_rng(0)
    papers = [
        {
            "id":        f"p{i}",
            "title":     f"Paper {i}",
            "year":      2015 + i,
            "citations": int(rng.integers(5, 100)),
            "features":  rng.standard_normal(64).tolist(),
        }
        for i in range(n)
    ]
    edges = [{"src": i, "dst": max(i-1,0), "year": papers[i]["year"]}
             for i in range(1, n)]
    return {"papers": papers, "edges": edges, "top_k": 3}


def test_rank_returns_results():
    r = client.post("/api/v1/rank", json=_sample_payload())
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    assert data["results"][0]["rank"] == 1


def test_rank_scores_are_sorted():
    r = client.post("/api/v1/rank", json=_sample_payload(10))
    results = r.json()["results"]
    scores  = [p["score"] for p in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_empty_papers():
    r = client.post("/api/v1/rank",
                    json={"papers": [], "edges": [], "top_k": 5})
    assert r.status_code == 400


def test_velocity_endpoint():
    r = client.post("/api/v1/velocity", json=_sample_payload())
    assert r.status_code == 200
    ts = r.json()["timeseries"]
    assert len(ts) > 0
    assert "year" in ts[0]
    assert "mean_velocity" in ts[0]
