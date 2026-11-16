import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
import torch.nn.functional as F
import torch.nn as nn
import os
import faiss


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=False, num_cols=2, figsize=(20, 10), title_fontsize=30, number = None, directory2save = './Results/'):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    [axi.set_axis_off() for axi in axes.ravel()]
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)
        # list_axes[i].axis('off')

    fig.tight_layout()

    #_ = plt.show()
    plt.savefig(directory2save + str(number) + '.png', pad_inches=0.0, bbox_inches='tight', transparent=True)
    fig.clf()
    plt.close('all')
    del fig
    gc.collect()

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    if isinstance(probFea, torch.Tensor):
        probFea = probFea.detach().cpu().numpy().astype(np.float32)
    if isinstance(galFea, torch.Tensor):
        galFea = galFea.detach().cpu().numpy().astype(np.float32)

    query_num = probFea.shape[0]
    gallery_num = galFea.shape[0]

    feat = np.concatenate([probFea, galFea], axis=0).astype(np.float32)
    all_num, feat_dim = feat.shape

    topk = 20
    topk = max(topk, k1 + 1)

    index = faiss.IndexFlatL2(feat_dim)
    index.add(feat)

    distances, indices = index.search(feat, topk)

    neighbor_sets = [set(indices[i]) for i in range(all_num)]

    V = [dict() for _ in range(all_num)]
    
    for i in range(all_num):
        forward_k_neigh_index = indices[i, :k1 + 1]
        reciprocal = []
        for candidate in forward_k_neigh_index:
            candidate_forward = indices[candidate, :k1 + 1]
            if i in candidate_forward:
                reciprocal.append(candidate)
        k_reciprocal_index = np.array(reciprocal)
        k_reciprocal_expansion_index = set(k_reciprocal_index)
        for candidate in k_reciprocal_index:
            candidate_forward = indices[candidate, :int(np.around(k1 / 2)) + 1]
            reciprocal_candidate = []
            for neighbor in candidate_forward:
                if candidate in indices[neighbor, :int(np.around(k1 / 2)) + 1]:
                    reciprocal_candidate.append(neighbor)
            reciprocal_candidate = set(reciprocal_candidate)
            if len(reciprocal_candidate) > 0:
                overlap = reciprocal_candidate.intersection(k_reciprocal_expansion_index)
                if len(overlap) > 2 / 3 * len(reciprocal_candidate):
                    k_reciprocal_expansion_index |= reciprocal_candidate
        k_reciprocal_expansion_index = list(k_reciprocal_expansion_index)
        dist_map = {}
        for idx, d in zip(indices[i], distances[i]):
            dist_map[idx] = d
        weights = []
        for j in k_reciprocal_expansion_index:
            d = dist_map.get(j, 1e6)
            weights.append(np.exp(-d))
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        for j, w in zip(k_reciprocal_expansion_index, weights):
            V[i][j] = w
    if k2 != 1:
        V_qe = [dict() for _ in range(all_num)]
        for i in range(all_num):
            neighbors = indices[i, :k2]
            agg = {}
            for n in neighbors:
                for k, v in V[n].items():
                    agg[k] = agg.get(k, 0) + v
            for k in agg:
                agg[k] /= k2
            V_qe[i] = agg
        V = V_qe
    invIndex = {}
    for i in range(all_num):
        for j in V[i]:
            if j not in invIndex:
                invIndex[j] = []
            invIndex[j].append(i)
    final_dist = np.full((query_num, gallery_num), 1e6, dtype=np.float32)
    for i in range(query_num):
        temp = {}
        for j, w_ij in V[i].items():
            if j not in invIndex:
                continue
            for k in invIndex[j]:
                w_kj = V[k].get(j, 0)
                val = min(w_ij, w_kj)
                temp[k] = temp.get(k, 0) + val
        for k, val in temp.items():
            jaccard = 1 - val / (2 - val)
            if k >= query_num:
                g = k - query_num
                original_d = 1e6
                for idx, d in zip(indices[i], distances[i]):
                    if idx == k:
                        original_d = d
                        break
                final = jaccard * (1 - lambda_value) + original_d * lambda_value
                final_dist[i, g] = final

    return final_dist

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
          
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)



# Graph construction, GCN refiner and graph re-ranking live in litegcr.py
# (shared, identical copy across TransReID / RPTM / MBR4G).
from litegcr import build_graphs, normalize_adj, GCNRefiner, graph_reranking, refine_with_gcn  # noqa: E402,F401
