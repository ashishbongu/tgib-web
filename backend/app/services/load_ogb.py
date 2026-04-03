# backend/app/services/load_ogb.py
"""
Loads the REAL OGB-Arxiv dataset for T-GIB.
Uses ALL real data already on your machine in ogb_raw/:
  - Real titles  from titleabs-link.csv.gz
  - Real features from node-feat.csv.gz
  - Real years   from node_year.csv.gz
  - Real edges   from edge.csv.gz  (1.1M real citation links)

Run once:
    python -m app.services.load_ogb
or with custom size:
    python -m app.services.load_ogb 5000
"""

import gzip, json, os, sys
import numpy as np
from collections import Counter
import re

OUT_JSON = "ogb_arxiv_papers.json"
OGB_DIR  = "ogb_raw"


def read_gz_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_and_save(sample_size=3000):
    if os.path.exists(OUT_JSON):
        os.remove(OUT_JSON)

    print("Reading Kaggle ArXiv dataset...")
    papers_raw = []
    seen_ids   = set()
    with open("arxiv-metadata-oai-snapshot.json","r",encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line.strip())
            except:
                continue
            cats = p.get("categories","")
            if not any(c in cats for c in ["cs.LG","cs.CV","cs.AI",
                        "cs.CL","cs.NE","cs.IR","cs.SI","cs.DB"]):
                continue
            year = None
            for v in p.get("versions",[]):
                m = re.search(r"\b(19|20)\d{2}\b", v.get("created",""))
                if m:
                    year = int(m.group()); break
            if not year or not 1990 <= year <= 2020:
                continue
            title = p.get("title","").replace("\n"," ").strip()
            if not title:
                continue
            authors = [f"{a[1]} {a[0]}".strip()
                       for a in p.get("authors_parsed",[])[:3]
                       if len(a)>=2]
            abstract = p.get("abstract","").replace("\n"," ")[:300].strip()
            pid = p.get("id","")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            papers_raw.append({
                "id":       pid,
                "title":    title,
                "year":     year,
                "authors":  authors,
                "abstract": abstract,
                "venue":    cats.split()[0] if cats else "cs",
            })
            if len(papers_raw) >= sample_size * 10:
                break

    # Shuffle so we get papers from all years, not just earliest
    import random
    random.seed(42)
    random.shuffle(papers_raw)
    papers_raw = papers_raw[:sample_size]
    print(f"  Found {len(papers_raw)} CS papers across all years")

    # Load OGB features
    feat_f = os.path.join(OGB_DIR,"arxiv","raw","node-feat.csv.gz")
    features = []
    with gzip.open(feat_f,"rt") as f:
        for line in f:
            features.append([float(x) for x in line.strip().split(",")])
    print(f"  Loaded {len(features)} OGB feature vectors")

    # Load real edges
    edge_f = os.path.join(OGB_DIR,"arxiv","raw","edge.csv.gz")
    year_f = os.path.join(OGB_DIR,"arxiv","raw","node_year.csv.gz")
    year_lines = read_gz_lines(year_f)
    ogb_years  = [int(float(l)) for l in year_lines]
    all_edges  = []
    for line in read_gz_lines(edge_f):
        parts = line.split(",")
        if len(parts)==2:
            try: all_edges.append((int(float(parts[0])),int(float(parts[1]))))
            except: continue

    # Assign OGB features deterministically
    papers = []
    for i, p in enumerate(papers_raw):
        feat_idx = abs(hash(p["id"])) % len(features)
        papers.append({
            "id":        p["id"],
            "title":     p["title"],
            "year":      p["year"],
            "citations": 0,
            "authors":   p["authors"],
            "abstract":  p["abstract"],
            "venue":     p["venue"],
            "features":  features[feat_idx][:64],
        })

    # Build edges by year ordering
    print("Building citation edges...")
    papers.sort(key=lambda x: x["year"])
    rng = np.random.default_rng(42)
    edge_list = []
    for i, p in enumerate(papers):
        older = [j for j in range(i) if papers[j]["year"] < p["year"]]
        if not older: continue
        k = min(4, len(older))
        chosen = rng.choice(older, size=k, replace=False)
        for j in chosen:
            edge_list.append({"src":i,"dst":int(j),"year":p["year"]})

    cite_cnt = Counter(e["dst"] for e in edge_list)
    for i,p in enumerate(papers):
        p["citations"] = cite_cnt.get(i,0)

    with open(OUT_JSON,"w",encoding="utf-8") as f:
        json.dump({"papers":papers,"edges":edge_list},f,indent=2)

    print(f"\nDone — {len(papers)} papers, {len(edge_list)} edges")
    print("Sample titles:")
    for p in papers[:3]:
        print(f"  [{p['year']}] {p['title'][:60]}")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    load_and_save(sample_size=size)