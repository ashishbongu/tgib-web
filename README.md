# T-GIB Web — Paradigm Shift Ranker

A full-stack web application that surfaces paradigm-shifting research papers
using the Temporal Graph Information Bottleneck (T-GIB) framework.

---

## Team Members

| Name | Roll Number | GitHub |
| :--- | :--- | :--- |
| BONGU ASHISH | 23BDS014 | [@ashishbongu](https://github.com/ashishbongu) |
| TARAN JAIN | 23BDS062 | [@Taranjain](https://github.com/Taranjain) |
| YASH BIRJE | 23BDS070 | [@Yash-Birje](https://github.com/Yash-Birje) |
| C SAI ARAVIND | 23BDS075 | [@comet-web](https://github.com/comet-web) |

---

## Project Structure

```
tgib-web/
├── backend/
│   ├── app/
│   │   ├── main.py              ← FastAPI app entry point
│   │   ├── api/routes.py        ← All REST API endpoints
│   │   ├── core/tgib_model.py   ← T-GIB inference engine
│   │   ├── models/schemas.py    ← Pydantic request/response schemas
│   │   └── services/
│   │       └── data_service.py  ← Semantic Scholar API + fallback data
│   ├── tests/test_api.py        ← Pytest tests
│   └── requirements.txt
├── frontend/
│   └── public/index.html        ← Single-page app (HTML + JS + Chart.js)
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   └── Dockerfile.frontend
│   └── k8s/deployment.yaml      ← Kubernetes manifests
├── .github/workflows/deploy.yml ← CI/CD pipeline
├── docker-compose.yml
└── scripts/setup_local.sh
```

---

## How to Run

### Option A — Docker (easiest, recommended)

**Requirements:** Docker Desktop installed and running.

```bash
# 1. Clone / download the project
git clone https://github.com/your-username/tgib-web.git
cd tgib-web

# 2. Start everything with one command
docker compose up --build

# 3. Open in browser
#    Frontend:  http://localhost:3000
#    API docs:  http://localhost:8000/docs
```

To stop: `docker compose down`

---

### Option B — Python only (no Docker)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start the backend
cd backend
uvicorn app.main:app --reload --port 8000

# 4. Open the frontend
# Just open frontend/public/index.html in your browser
# (double-click the file, or use VS Code Live Server extension)
```

API docs available at: http://localhost:8000/docs

---

### Option C — Run tests

```bash
source .venv/bin/activate
pytest backend/tests/ -v
```

---

## REST API Reference

| Method | Endpoint                        | Description                              |
|--------|---------------------------------|------------------------------------------|
| GET    | `/api/v1/health`                | Health check                             |
| POST   | `/api/v1/search`                | Search by topic + rank with T-GIB        |
| POST   | `/api/v1/rank`                  | Rank a provided list of papers           |
| POST   | `/api/v1/velocity`              | Get year-by-year velocity timeseries     |
| GET    | `/api/v1/paper/{id}/velocity`   | Velocity timeseries for a single paper   |

Full interactive docs: http://localhost:8000/docs

### Example: Search and rank

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "attention mechanism transformer",
    "top_k": 10,
    "year_from": 2015,
    "year_to": 2024
  }'
```

### Example: Rank your own papers

```bash
curl -X POST http://localhost:8000/api/v1/rank \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [
      {"id":"p1","title":"Attention Is All You Need","year":2017,
       "citations":50000,"features":[0.1,0.2,...]},
      {"id":"p2","title":"BERT","year":2018,
       "citations":30000,"features":[0.3,0.1,...]}
    ],
    "edges": [{"src":1,"dst":0,"year":2018}],
    "top_k": 5
  }'
```

---

## Cloud Deployment (AWS / GCP / Azure)

### Step 1 — Deploy to any cloud VM

```bash
# On your cloud VM (Ubuntu 22.04)
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
git clone https://github.com/your-username/tgib-web.git
cd tgib-web
docker compose up -d --build
```

The app is now live on your VM's public IP:
- Frontend: `http://<your-ip>:3000`
- API:      `http://<your-ip>:8000`

### Step 2 — Kubernetes (production scale)

```bash
# 1. Build and push images to a registry
docker build -f infra/docker/Dockerfile.backend -t your-registry/tgib-backend:latest .
docker build -f infra/docker/Dockerfile.frontend -t your-registry/tgib-frontend:latest .
docker push your-registry/tgib-backend:latest
docker push your-registry/tgib-frontend:latest

# 2. Edit infra/k8s/deployment.yaml — replace "your-registry" with your actual registry

# 3. Apply to Kubernetes cluster
kubectl apply -f infra/k8s/deployment.yaml

# 4. Get the public IP
kubectl get service tgib-frontend-svc
```

### Step 3 — Automatic CI/CD (GitHub Actions)

1. Push code to GitHub
2. Add these secrets in GitHub → Settings → Secrets:
   - `SERVER_HOST` — your VM IP address
   - `SERVER_USER` — SSH username (e.g. `ubuntu`)
   - `SERVER_SSH_KEY` — your private SSH key
3. Every push to `main` will automatically test, build, and deploy

---

## Cloud Concepts Used

| Concept            | Where                                      |
|--------------------|--------------------------------------------|
| Virtualisation     | Docker containers isolate each service     |
| Microservices      | Backend and frontend are separate services |
| REST API           | FastAPI exposes clean HTTP endpoints       |
| Container Registry | Docker images stored in GHCR               |
| Orchestration      | Kubernetes manages containers at scale     |
| Auto-scaling       | HPA scales backend pods under CPU load     |
| CI/CD Pipeline     | GitHub Actions: test → build → deploy      |
| Health checks      | Docker and Kubernetes monitor `/health`    |

---

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, NumPy
- **Frontend:** Plain HTML/CSS/JS, Chart.js
- **Containerisation:** Docker, docker-compose
- **Orchestration:** Kubernetes
- **CI/CD:** GitHub Actions
- **Data source:** Semantic Scholar API (falls back to synthetic data offline)
