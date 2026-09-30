from __future__ import annotations

import numpy as np


def kmeans_one_step_numpy(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    assign = dists.argmin(axis=1)
    new = np.stack(
        [
            points[assign == k].mean(axis=0) if (assign == k).any() else centers[k]
            for k in range(len(centers))
        ]
    )
    return new


def kmeans_one_step_torch(points, centers):
    import torch

    dists = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
    assign = dists.argmin(dim=1)
    new = torch.stack(
        [
            points[assign == k].mean(dim=0) if (assign == k).any() else centers[k]
            for k in range(centers.shape[0])
        ]
    )
    return new
