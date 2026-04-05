# backend/app/api/routes.py
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.deps import get_current_user
from app.core.tgib_model import TGIBModel
from app.models.schemas import (
    HealthResponse,
    RankRequest,
    RankResponse,
    RankedPaper,
    SearchRequest,
    VelocityPoint,
    VelocityResponse,
)
from app.services.data_service import build_citation_edges, fetch_papers_by_query

router = APIRouter()

# Single model instance (loaded once at startup)
_model: Optional[TGIBModel] = None


def get_model() -> TGIBModel:
    global _model
    if _model is None:
        _model = TGIBModel()
    return _model


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(status="ok", version="1.0.0", model="T-GIB")


@router.post("/rank", response_model=RankResponse, tags=["ranking"])
def rank_papers(req: RankRequest, _: dict = Depends(get_current_user)):
    """
    Rank a list of papers by T-GIB innovation velocity score.
    """
    if len(req.papers) == 0:
        raise HTTPException(status_code=400, detail="No papers provided.")

    model = get_model()
    papers = [p.model_dump() for p in req.papers]
    edges = [(e.src, e.dst, e.year) for e in req.edges]
    ranked = model.rank(papers, edges, top_k=req.top_k)

    return RankResponse(
        results=[RankedPaper(**r) for r in ranked],
        total=len(ranked),
    )


@router.post("/search", response_model=RankResponse, tags=["ranking"])
async def search_and_rank(req: SearchRequest, _: dict = Depends(get_current_user)):
    """
    Search the bundled paper corpus by query string, then rank results with T-GIB.
    Falls back to synthetic data if the dataset file is unavailable.
    """
    from app.services.data_service import _synthetic_papers, load_ogb_papers

    try:
        all_papers, all_edges = load_ogb_papers()
    except FileNotFoundError:
        all_papers = _synthetic_papers(req.query, max(req.top_k * 5, 50))
        all_edges = []

    papers = [
        p for p in all_papers
        if req.year_from <= p["year"] <= req.year_to
    ]

    query_clean = req.query.strip().lower()
    if query_clean and query_clean != ".":
        words = [w for w in query_clean.split() if len(w) > 2]
        if words:
            def score_paper(paper):
                text = (
                    paper.get("title", "") + " " +
                    paper.get("abstract", "") + " " +
                    paper.get("venue", "")
                ).lower()
                return sum(1 for w in words if w in text)

            matched = [p for p in papers if score_paper(p) > 0]
            if len(matched) >= 3:
                matched.sort(key=score_paper, reverse=True)
                papers = matched[:200]

    if len(papers) == 0:
        papers = all_papers
        words = [w for w in query_clean.split() if len(w) > 2]
        if words:
            matched = [
                p for p in papers
                if any(
                    w in (p.get("title", "") + " " + p.get("abstract", "")).lower()
                    for w in words
                )
            ]
            if matched:
                papers = matched

    if not papers:
        raise HTTPException(status_code=404, detail=f"No papers found for: {req.query}")

    papers = [
        p for p in papers
        if req.year_from <= p["year"] <= req.year_to
    ]
    if not papers:
        raise HTTPException(status_code=404, detail=f"No papers found for: {req.query}")

    cite_counter = defaultdict(int)
    for edge in all_edges:
        cite_counter[str(edge["dst"])] += 1

    hydrated_papers = []
    for paper in papers:
        hydrated = paper.copy()
        if hydrated["citations"] == 0:
            hydrated["citations"] = cite_counter.get(str(hydrated["id"]), 0)
        hydrated_papers.append(hydrated)

    edges = build_citation_edges(hydrated_papers)
    model = get_model()
    ranked = model.rank(hydrated_papers, edges, top_k=req.top_k)

    return RankResponse(
        results=[RankedPaper(**r) for r in ranked],
        total=len(ranked),
    )


@router.post("/velocity", response_model=VelocityResponse, tags=["analytics"])
def velocity_timeseries(req: RankRequest, _: dict = Depends(get_current_user)):
    """
    Return year-by-year mean innovation velocity for a set of papers.
    """
    model = get_model()
    papers = [p.model_dump() for p in req.papers]
    edges = [(e.src, e.dst, e.year) for e in req.edges]
    ts = model.velocity_timeseries(papers, edges)

    return VelocityResponse(
        timeseries=[
            VelocityPoint(year=yr, **vals)
            for yr, vals in sorted(ts.items())
        ]
    )


@router.get("/paper/{paper_id}/velocity", tags=["analytics"])
async def paper_velocity(
    paper_id: str,
    query: str = Query(..., description="Topic query to build context graph"),
    _: dict = Depends(get_current_user),
):
    """
    Return the velocity timeseries for a single paper identified by ID.
    """
    papers = await fetch_papers_by_query(query, limit=30)
    target = next((p for p in papers if p["id"] == paper_id), None)

    if not target:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found.")

    edges = build_citation_edges(papers)
    model = get_model()
    ts = model.velocity_timeseries(papers, edges)

    return {
        "paper_id": paper_id,
        "title": target["title"],
        "timeseries": ts,
    }
