# backend/app/core/tgib_model.py
"""
T-GIB Inference Engine
Loads pre-trained weights and runs ranking on citation graph snapshots.
"""

import numpy as np
import pickle
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def relu(x):     return np.maximum(0, x)
def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def softmax(x):  e = np.exp(x - x.max()); return e / (e.sum() + 1e-12)
def xavier(fi, fo, rng):
    lim = np.sqrt(6 / (fi + fo))
    return rng.uniform(-lim, lim, (fi, fo)).astype(np.float64)


class TGNNEncoder:
    def __init__(self, feat_dim: int, hidden_dim: int, rng):
        self.W1 = xavier(feat_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier(hidden_dim, hidden_dim, rng)
        self.b2 = np.zeros(hidden_dim)
        self.Wa = xavier(hidden_dim, 1, rng)

    def encode(self, node: int, adj: dict, feats: np.ndarray) -> np.ndarray:
        h = relu(feats[node].astype(np.float64) @ self.W1 + self.b1)
        nbs = adj.get(node, [])[:20]
        if nbs:
            Hn  = np.stack([relu(feats[nb].astype(np.float64) @ self.W1 + self.b1)
                             for nb in nbs])
            at  = softmax(np.atleast_1d((Hn @ self.Wa).squeeze()))
            h   = 0.5 * h + 0.5 * (at[:, None] * Hn).sum(0)
        return relu(h @ self.W2 + self.b2).astype(np.float32)


class InformationBottleneck:
    def __init__(self, hidden_dim: int, latent_dim: int,
                 beta: float, lr: float, rng):
        self.beta = beta
        self.lr   = lr
        self.rng  = rng
        mid = hidden_dim // 2 + latent_dim
        self.W1  = xavier(hidden_dim, mid, rng);  self.b1  = np.zeros(mid)
        self.Wmu = xavier(mid, latent_dim, rng);  self.bmu = np.zeros(latent_dim)
        self.Wls = xavier(mid, latent_dim, rng);  self.bls = np.zeros(latent_dim)
        self.Wd1 = xavier(latent_dim, mid, rng);  self.bd1 = np.zeros(mid)
        self.Wd2 = xavier(mid, 1, rng);           self.bd2 = np.zeros(1)
        pkeys = 'W1 b1 Wmu bmu Wls bls Wd1 bd1 Wd2 bd2'.split()
        self._m = {p: np.zeros_like(getattr(self, p)) for p in pkeys}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in pkeys}
        self._t = 0

    def encode(self, h: np.ndarray, train: bool = False):
        h  = np.asarray(h, np.float64)
        h1 = relu(h @ self.W1 + self.b1)
        mu = h1 @ self.Wmu + self.bmu
        ls = np.clip(h1 @ self.Wls + self.bls, -4, 2)
        sig = np.exp(ls)
        eps = self.rng.standard_normal(mu.shape) if train else np.zeros_like(mu)
        return (mu + sig * eps).astype(np.float32), mu, sig

    def get_latent(self, h: np.ndarray) -> np.ndarray:
        z, _, _ = self.encode(h, train=False)
        return z

    def decode(self, z: np.ndarray) -> float:
        z  = np.asarray(z, np.float64)
        d1 = relu(z @ self.Wd1 + self.bd1)
        return float(sigmoid((d1 @ self.Wd2 + self.bd2)[0]))

    def train_step(self, h: np.ndarray, y: int) -> Tuple[float, np.ndarray]:
        z, mu, sig, = self.encode(h, train=True)
        # full backprop omitted here — use saved weights in production
        return 0.0, z


class TGIBModel:
    """
    Main T-GIB model. Loads weights from disk or initialises fresh.
    Call .rank(papers, edges) to get sorted results.
    """

    WEIGHTS_PATH = os.environ.get("TGIB_WEIGHTS", "tgib_weights.pkl")

    def __init__(self, feat_dim: int = 64, hidden_dim: int = 64,
                 latent_dim: int = 32, beta: float = 0.001,
                 alpha: float = 0.6, seed: int = 42):
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta       = beta
        self.alpha      = alpha
        rng             = np.random.default_rng(seed)

        self.encoder = TGNNEncoder(feat_dim, hidden_dim, rng)
        self.ib      = InformationBottleneck(hidden_dim, latent_dim,
                                              beta, 3e-4, rng)

        if os.path.exists(self.WEIGHTS_PATH):
            self._load_weights()

    # ── Weight persistence ───────────────────────────────────────────────

    def save_weights(self, path: Optional[str] = None):
        path = path or self.WEIGHTS_PATH
        state = {
            "encoder": {k: getattr(self.encoder, k)
                        for k in ['W1','b1','W2','b2','Wa']},
            "ib":      {k: getattr(self.ib, k)
                        for k in ['W1','b1','Wmu','bmu','Wls','bls',
                                   'Wd1','bd1','Wd2','bd2']},
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def _load_weights(self):
        with open(self.WEIGHTS_PATH, 'rb') as f:
            state = pickle.load(f)
        for k, v in state["encoder"].items():
            setattr(self.encoder, k, v)
        for k, v in state["ib"].items():
            setattr(self.ib, k, v)

    # ── Core ranking ─────────────────────────────────────────────────────

    def rank(self,
             papers: List[Dict],
             edges:  List[Tuple[int, int, int]],
             top_k:  int = 20) -> List[Dict]:
        """
        papers : [{"id": str, "title": str, "year": int,
                    "features": List[float], "citations": int}, ...]
        edges  : [(src_idx, dst_idx, year), ...]
        Returns papers sorted by T-GIB score, top_k results.
        """
        n     = len(papers)
        feats = np.stack([np.array(p["features"], dtype=np.float32)
                          for p in papers])
        years = np.array([p["year"] for p in papers])
        cites = np.array([p.get("citations", 0) for p in papers], dtype=float)

        # Build temporal adjacency
        adj_all = defaultdict(list)
        for s, t, yr in edges:
            adj_all[t].append((s, yr))

        # Run T-GIB year by year
        H_prev  = {}
        V_all   = np.zeros(n)
        Z_final = {}

        for yr in sorted(set(years)):
            adj_yr = defaultdict(list)
            for t_node, entries in adj_all.items():
                adj_yr[t_node] = [s for s, ey in entries if ey <= yr]

            pub = np.where(years <= yr)[0]
            for node in pub:
                H_v  = self.encoder.encode(node, adj_yr, feats)
                mu   = self.ib.get_latent(H_v)
                Z_final[node] = mu
                if node in H_prev:
                    V_all[node] = float(np.linalg.norm(mu - H_prev[node]))
                H_prev[node] = mu.copy()

        # Composite score
        S = self.alpha * V_all + (1 - self.alpha) * np.log1p(cites)
        S = S / (S.max() + 1e-9)

        # Build result
        ranked_idx = np.argsort(S)[::-1][:top_k]
        results = []
        for rank_pos, idx in enumerate(ranked_idx, 1):
            p = papers[idx].copy()
            p["rank"]       = rank_pos
            p["score"]      = float(S[idx])
            p["velocity"]   = float(V_all[idx])
            p["latent"]     = Z_final.get(idx, np.zeros(self.latent_dim)).tolist()
            results.append(p)
        return results

    def velocity_timeseries(self,
                             papers: List[Dict],
                             edges:  List[Tuple[int, int, int]]) -> Dict:
        """Returns per-year mean velocity for shift vs normal papers."""
        n     = len(papers)
        feats = np.stack([np.array(p["features"], dtype=np.float32)
                          for p in papers])
        years = np.array([p["year"] for p in papers])

        adj_all = defaultdict(list)
        for s, t, yr in edges:
            adj_all[t].append((s, yr))

        H_prev = {}
        year_data = {}

        for yr in sorted(set(years)):
            adj_yr = defaultdict(list)
            for t_node, entries in adj_all.items():
                adj_yr[t_node] = [s for s, ey in entries if ey <= yr]

            pub = np.where(years <= yr)[0]
            velocities = []
            for node in pub:
                H_v = self.encoder.encode(node, adj_yr, feats)
                mu  = self.ib.get_latent(H_v)
                if node in H_prev:
                    velocities.append(float(np.linalg.norm(mu - H_prev[node])))
                H_prev[node] = mu.copy()

            year_data[int(yr)] = {
                "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
                "n_papers":      int(len(pub))
            }

        return year_data
