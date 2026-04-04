# backend/app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Paper(BaseModel):
    id:         str
    title:      str
    year:       int
    citations:  int                = 0
    features:   List[float]        = Field(default_factory=list)
    abstract:   Optional[str]      = None
    authors:    Optional[List[str]]= None
    venue:      Optional[str]      = None


class Edge(BaseModel):
    src:  int
    dst:  int
    year: int


class RankRequest(BaseModel):
    papers: List[Paper]
    edges:  List[Edge]
    top_k:  int = 20


class RankedPaper(BaseModel):
    id:         str
    title:      str
    year:       int
    citations:  int
    rank:       int
    score:      float
    velocity:   float
    latent:     List[float]
    abstract:   Optional[str]      = None
    authors:    Optional[List[str]]= None
    venue:      Optional[str]      = None


class RankResponse(BaseModel):
    results:     List[RankedPaper]
    total:       int
    model:       str = "T-GIB v1.0"


class VelocityPoint(BaseModel):
    year:          int
    mean_velocity: float
    n_papers:      int


class VelocityResponse(BaseModel):
    timeseries: List[VelocityPoint]


class HealthResponse(BaseModel):
    status:  str
    version: str
    model:   str


class SearchRequest(BaseModel):
    query:     str
    year_from: Optional[int] = 2013
    year_to:   Optional[int] = 2024
    top_k:     int           = 20


class ErrorResponse(BaseModel):
    detail: str
    code:   int


class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type:   str
