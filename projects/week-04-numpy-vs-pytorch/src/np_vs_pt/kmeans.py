from __future__ import annotations
import numpy as np


def kmeans_one_step_numpy(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    assign = dists.argmin(axis=1)
    new = np.stack([X[assign == k].mean(axis=0) if (assign == k).any() else centers[k]
                    for k in range(len(centers))])
    return new


def kmeans_one_step_torch(X, centers):
    import torch
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
    assign = dists.argmin(dim=1)
    new = torch.stack([X[assign == k].mean(dim=0) if (assign == k).any() else centers[k]
                       for k in range(centers.shape[0])])
    return new
