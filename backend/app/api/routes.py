# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.schemas import (
    RankRequest, RankResponse, RankedPaper,
    VelocityResponse, VelocityPoint,
    HealthResponse, SearchRequest
)
from app.services.data_service import (
    fetch_papers_by_query, build_citation_edges
)
from app.core.tgib_model import TGIBModel

router = APIRouter()

# Single model instance (loaded once at startup)
_model: Optional[TGIBModel] = None




def get_model() -> TGIBModel:
    global _model
    if _model is None:
        _model = TGIBModel()
    return _model


# ── Health ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(status="ok", version="1.0.0", model="T-GIB")


# ── Rank papers ───────────────────────────────────────────────────────────

@router.post("/rank", response_model=RankResponse, tags=["ranking"])
def rank_papers(req: RankRequest):
    """
    Rank a list of papers by T-GIB innovation velocity score.

    Send papers with features + citation edges.
    Returns top_k papers sorted by paradigm-shift score.
    """
    if len(req.papers) == 0:
        raise HTTPException(status_code=400, detail="No papers provided.")

    model  = get_model()
    papers = [p.model_dump() for p in req.papers]
    edges  = [(e.src, e.dst, e.year) for e in req.edges]

    ranked = model.rank(papers, edges, top_k=req.top_k)

    return RankResponse(
        results=[RankedPaper(**r) for r in ranked],
        total=len(ranked)
    )


# ── Search + auto-rank ────────────────────────────────────────────────────

@router.post("/search", response_model=RankResponse, tags=["ranking"])
async def search_and_rank(req: SearchRequest):
    """
    Search Semantic Scholar by query string, then rank results with T-GIB.
    Falls back to synthetic data when offline.
    """
    from app.services.data_service import load_ogb_papers
    all_papers, all_edges = load_ogb_papers()

    # Filter by year range and sample 200 papers around the query
    import random
    papers = [p for p in all_papers
            if req.year_from <= p["year"] <= req.year_to]
    
    papers = random.sample(papers, min(200, len(papers)))

    # Count citations from edges
    from collections import defaultdict
    cite_counter = defaultdict(int)
    for e in all_edges:
        cite_counter[str(e["dst"])] += 1

    for p in papers:
        if p["citations"] == 0:          # only override if not already set
            p["citations"] = cite_counter.get(p["id"], 0)

    # Get corresponding edges
    paper_ids = {p["id"] for p in papers}
    idx_map   = {p["id"]: i for i, p in enumerate(papers)}
    edges = []
    for e in all_edges:
        s, d = str(e["src"]), str(e["dst"])
        if s in paper_ids and d in paper_ids:
            edges.append({"src": idx_map[s], "dst": idx_map[d],
                        "year": e["year"]})

    if not papers:
        raise HTTPException(status_code=404,
                             detail=f"No papers found for: {req.query}")

    # Filter by year range
    papers = [p for p in papers
              if req.year_from <= p["year"] <= req.year_to]

    if not papers:
        raise HTTPException(status_code=404,
                             detail="No papers in the given year range.")

    edges  = build_citation_edges(papers)
    model  = get_model()
    ranked = model.rank(papers, edges, top_k=req.top_k)

    return RankResponse(
        results=[RankedPaper(**r) for r in ranked],
        total=len(ranked)
    )


# ── Velocity timeseries ───────────────────────────────────────────────────

@router.post("/velocity", response_model=VelocityResponse, tags=["analytics"])
def velocity_timeseries(req: RankRequest):
    """
    Return year-by-year mean innovation velocity for a set of papers.
    Used to draw the timestamp velocity chart in the frontend.
    """
    model  = get_model()
    papers = [p.model_dump() for p in req.papers]
    edges  = [(e.src, e.dst, e.year) for e in req.edges]

    ts = model.velocity_timeseries(papers, edges)

    return VelocityResponse(
        timeseries=[
            VelocityPoint(year=yr, **vals)
            for yr, vals in sorted(ts.items())
        ]
    )


# ── Single paper velocity ─────────────────────────────────────────────────

@router.get("/paper/{paper_id}/velocity", tags=["analytics"])
async def paper_velocity(
    paper_id: str,
    query:    str = Query(..., description="Topic query to build context graph")
):
    """
    Return the velocity timeseries for a single paper identified by ID.
    Builds context by fetching related papers from Semantic Scholar.
    """
    papers = await fetch_papers_by_query(query, limit=30)
    target = next((p for p in papers if p["id"] == paper_id), None)

    if not target:
        raise HTTPException(status_code=404,
                             detail=f"Paper {paper_id} not found.")

    edges  = build_citation_edges(papers)
    model  = get_model()
    ts     = model.velocity_timeseries(papers, edges)

    return {"paper_id": paper_id, "title": target["title"],
            "timeseries": ts}
