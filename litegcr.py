"""
LiteGCR - shared implementation (identical copy in TransReID, RPTM and MBR4G).

  * Global graph      (Algorithm 1): kNN over L2-normalised features.
  * Cross-camera graph (Algorithm 2): kNN restricted to samples from a
    different camera. If camera IDs are unavailable (e.g. VehicleID) the
    cross-camera branch is disabled: A_c = None and alpha is fixed to 1.
  * The same graph builder is used for training batches and for the
    query+gallery graph at test time, so train/test graphs are consistent.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_K = 20
DEFAULT_GAMMA = 0.5
DEFAULT_ALPHA = 0.8  # fixed mixing weight, rule-based graph re-ranking only


def _as_index_tensor(x, device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device).view(-1)
    return torch.as_tensor(np.asarray(x), device=device).view(-1)


def _knn_adjacency(dist, k, gamma):
    """Row i keeps its k smallest finite entries of `dist` (inf = not allowed),
    weights exp(-d^2 / gamma), row-normalised. Rows with fewer than k allowed
    neighbours keep only those (k_i = min(k, #allowed)). Returns sparse COO."""
    n = dist.size(0)
    k = min(k, n)
    vals, idx = torch.topk(dist, k, dim=1, largest=False)
    w = torch.exp(-(vals ** 2) / gamma)
    w = torch.where(torch.isfinite(vals), w, torch.zeros_like(w))
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

    rows = torch.arange(n, device=dist.device).repeat_interleave(k)
    cols = idx.reshape(-1)
    w = w.reshape(-1)
    keep = w > 0
    return torch.sparse_coo_tensor(
        torch.stack([rows[keep], cols[keep]]), w[keep], (n, n)
    ).coalesce()


def normalize_adj(adj):
    """D^-1/2 (A + I) D^-1/2 for a sparse adjacency matrix."""
    n = adj.size(0)
    device = adj.device
    diag = torch.arange(n, device=device)
    indices = torch.cat([adj.indices(), torch.stack([diag, diag])], dim=1)
    values = torch.cat([adj.values(), torch.ones(n, device=device, dtype=adj.dtype)])
    a_hat = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    deg = torch.sparse.sum(a_hat, dim=1).to_dense()
    d_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    i = a_hat.indices()
    return torch.sparse_coo_tensor(
        i, a_hat.values() * d_inv_sqrt[i[0]] * d_inv_sqrt[i[1]], (n, n)
    ).coalesce()


@torch.no_grad()
def build_graphs(feat, camids=None, k=DEFAULT_K, gamma=DEFAULT_GAMMA):
    """Build the normalised global graph A_g and cross-camera graph A_c.

    feat:   (N, D) features (gradients are not propagated through the graph).
    camids: (N,) camera IDs, or None when the dataset has no camera labels.
    Returns (A_g, A_c); A_c is None when camids is None.
    """
    x = F.normalize(feat.detach().float(), p=2, dim=1)
    dist = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))  # self-loop is added by normalize_adj

    A_g = normalize_adj(_knn_adjacency(dist, k, gamma))
    if camids is None:
        return A_g, None

    cam = _as_index_tensor(camids, x.device)
    dist.masked_fill_(cam.view(-1, 1) == cam.view(1, -1), float("inf"))
    A_c = normalize_adj(_knn_adjacency(dist, k, gamma))
    return A_g, A_c


class GCNRefiner(nn.Module):
    def __init__(self, feature_dim):
        super(GCNRefiner, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
        nn.init.kaiming_uniform_(self.W, a=0.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.bn = nn.LayerNorm(feature_dim)
        self.relu = nn.ReLU()

    def _forward(self, features, A_g, A_c):
        x = features.float()
        if A_c is None:  # no camera labels: alpha fixed to 1
            support = torch.sparse.mm(A_g, x)
        else:
            alpha = torch.clamp(self.alpha, 0.0, 1.0)
            support = alpha * torch.sparse.mm(A_g, x) + (1 - alpha) * torch.sparse.mm(A_c, x)
        out = self.relu(self.bn(torch.mm(support, self.W)))
        return x + out

    def forward(self, features, A_g, A_c=None):
        if features.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward(features, A_g, A_c)
        return self._forward(features, A_g, A_c)

    def load_param(self, trained_path):
        if os.path.exists(trained_path):
            state_dict = torch.load(trained_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict)
            print(f"==> GCN model loaded from {trained_path}")
        else:
            print(f"==> No GCN checkpoint found at {trained_path}, training from scratch.")


def refine_with_gcn(features, camids, gcn_model, k=DEFAULT_K, gamma=DEFAULT_GAMMA, normalize_rest=True):
    """Apply LiteGCR to the global part of `features` (first W.shape[0] dims).
    The graph is built on that global part, as during training. Any remaining
    dims (e.g. JPM local features, other MBR branches) are passed through."""
    d = gcn_model.W.shape[0]
    g, rest = features[:, :d], features[:, d:]
    A_g, A_c = build_graphs(g, camids, k, gamma)
    g = F.normalize(gcn_model(g, A_g, A_c), p=2, dim=1)
    if rest.numel() == 0:
        return g
    if normalize_rest:
        rest = F.normalize(rest, p=2, dim=1)
    return torch.cat([g, rest.float()], dim=1)


@torch.no_grad()
def graph_reranking(probFea, galFea, q_camids, g_camids, k=DEFAULT_K, gamma=DEFAULT_GAMMA,
                    alpha=DEFAULT_ALPHA, learn_based=True, gcn_model=None, use_camera=True,
                    normalize_rest=True):
    """Query-gallery distance after graph refinement.

    gcn_model given (and learn_based): LiteGCR refinement (learned).
    otherwise: rule-based propagation alpha*A_g X + (1-alpha)*A_c X.
    use_camera=False (no camera labels): A_c disabled, alpha = 1.
    Returns a numpy (num_query, num_gallery) distance matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = probFea.to(device).float()
    g = galFea.to(device).float()
    feats = torch.cat([q, g], dim=0)

    cams = None
    if use_camera:
        cams = torch.cat([_as_index_tensor(q_camids, device), _as_index_tensor(g_camids, device)])

    if gcn_model is not None and learn_based:
        was_training = gcn_model.training
        gcn_model.to(device).eval()
        refined = refine_with_gcn(feats, cams, gcn_model, k, gamma, normalize_rest)
        gcn_model.train(was_training)
    else:
        A_g, A_c = build_graphs(feats, cams, k, gamma)
        x = F.normalize(feats, p=2, dim=1)
        if A_c is None:
            refined = torch.sparse.mm(A_g, x)
        else:
            refined = alpha * torch.sparse.mm(A_g, x) + (1 - alpha) * torch.sparse.mm(A_c, x)

    nq = q.size(0)
    distmat = torch.cdist(refined[:nq], refined[nq:], p=2)
    return distmat.cpu().numpy()


# Backwards-compatible alias used by RPTM.
graph_reranking_func = graph_reranking
