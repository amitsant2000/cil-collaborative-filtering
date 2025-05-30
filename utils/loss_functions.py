import torch
from ripser import ripser
import numpy as np

import torch.nn.functional as F

# from torchph.pershom import vr_persistence

def first_principle_component(s_latent, p_latent, ratings):
    total = 0

    unique_ratings = ratings.unique()
    total = 0

    for rating in unique_ratings:
        mask = ratings == rating
        s = s_latent[mask]
        p = p_latent[mask]
        if s.size(0) < 2 or p.size(0) < 2:
            continue
        s_pc = s @ torch.pca_lowrank(s, q=3)[2]
        p_pc = p @ torch.pca_lowrank(p, q=3)[2]
        total += torch.cosine_similarity(p_pc, s_pc).mean()


    return 1 - total / len(ratings.unique())


def info_nce_loss(s_latent, p_latent, ratings, threshold=4.0, temperature=0.07):
    pos_mask = ratings >= threshold
    s_pos = s_latent[pos_mask]
    p_pos = p_latent[pos_mask]
    
    if len(s_pos) < 2:  # Not enough for contrastive batch
        return torch.tensor(0.0, device=s_latent.device)

    logits = torch.matmul(s_pos, p_pos.T) / temperature
    labels = torch.arange(len(s_pos), device=logits.device)
    return F.cross_entropy(logits, labels)

def neg_info_nce_loss(s_latent, p_latent, ratings, threshold=2.0, temperature=0.07):
    neg_mask = ratings <= threshold
    s_pos = s_latent[neg_mask]
    p_pos = p_latent[neg_mask]
    
    if len(s_pos) < 2:  # Not enough for contrastive batch
        return torch.tensor(0.0, device=s_latent.device)

    logits = -torch.matmul(s_pos, p_pos.T) / temperature
    labels = torch.arange(len(s_pos), device=logits.device)
    return F.cross_entropy(logits, labels)

def fast_topological_loss(X, Z, ratings, n_sample=400):
    # X, Z: torch.Tensor, shape (batch_size, dim)
    # Subsample n_sample points from each batch for speed
    n_points = min(X.size(0), n_sample)
    idx = torch.randperm(X.size(0))[:n_points]

    
    X_sub = X[idx]
    Z_sub = Z[idx]
    ratings_sub = ratings[idx]

    combined = torch.cat([X_sub, Z_sub], dim=0)

    # print(f"Combined shape: {combined.shape}, Ratings shape: {ratings_sub.shape}")
    # return
    
    combined_dist = torch.cdist(combined, combined, p=2)
    ratings_dist = torch.fill(torch.empty_like(combined_dist), 501).to(combined_dist.device)

    # print(f"Combined distance matrix shape: {combined_dist.shape}, Ratings distance matrix shape: {ratings_dist.shape}")

    # Set the distances for the points corresponding to idx in the first half and idx + n_points in the second half
    # Vectorized version of setting distances
    i = torch.arange(n_points, device=combined_dist.device)
    noise = torch.rand(n_points, device=combined_dist.device) * 0.1
    ratings_dist[i, i + n_points] = 5 - ratings_sub + noise

    # Only compute H_0 (connected components)
    dgm_combined = vr_persistence(combined_dist, 0, 500)[0][0]
    dgm_ratings = vr_persistence(ratings_dist, 0, 500)[0][0]
    
    # print(f"Computed persistence diagrams: X={dgm_X.shape}, Z={dgm_Z.shape}")

    # Filter out infinite death times (points that never merge)
    dgm_combined = dgm_combined[torch.isfinite(dgm_combined[:, 1])]
    dgm_ratings = dgm_ratings[torch.isfinite(dgm_ratings[:, 1])]
    
    # Compute top-k lifetimes (death - birth)
    pers_combined = dgm_combined[:, 1] - dgm_combined[:, 0]
    pers_ratings = dgm_ratings[:, 1] - dgm_ratings[:, 0]
    
    # Pad with zeros so arrays are the same length
    def pad(arr, k):
        return F.pad(arr, (0, max(0, k - len(arr))), value=0)
    pers_combined = pad(pers_combined, max(len(pers_combined), len(pers_ratings)))
    pers_ratings = pad(pers_ratings, max(len(pers_combined), len(pers_ratings)))
    
    # L2 loss between lifetimes
    loss = F.mse_loss(pers_combined, pers_ratings)
    return loss