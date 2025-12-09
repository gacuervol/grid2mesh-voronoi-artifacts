import numpy as np
from typing import Callable, Optional
import random
import math

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
        """
        Select k points via farthest point sampling:
        - Start with a random initial point.
        - Iteratively add the point farthest from the current set.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
            Candidate point coordinates.
        k : int
            Number of points to select.

        Returns
        -------
        np.ndarray, shape (min(k, N), 2)
            Coordinates of selected points.
        """
        n = points.shape[0]
        if k <= 0 or n == 0:
            return np.empty((0, points.shape[1]))

        # Pick a random first index
        selected_idxs = [int(np.random.randint(n))]
        # Compute initial distances to first point
        dist = np.linalg.norm(points - points[selected_idxs[0]], axis=1)

        # Select remaining points
        for _ in range(1, min(k, n)):
            # Next point is the one with maximum minimum-distance
            next_idx = int(np.argmax(dist))
            selected_idxs.append(next_idx)
            # Update distances
            new_dist = np.linalg.norm(points - points[next_idx], axis=1)
            dist = np.minimum(dist, new_dist)

        return points[selected_idxs]


def weighted_farthest_point_sampling(points: np.ndarray, k: int, weights: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if k <= 0 or n == 0:
        return np.empty((0,2))

    # Get a point with probability proportional to the weights
    idx0 = np.random.choice(n, p=weights.ravel())
    selected = [idx0]
    dist = np.linalg.norm(points - points[idx0], axis=1)

    for _ in range(1, min(k, n)):
        score     = dist * weights
        next_idx  = int(np.argmax(score))
        selected.append(next_idx)
        # update distances
        new_dist  = np.linalg.norm(points - points[next_idx], axis=1)
        dist      = np.minimum(dist, new_dist)

    return points[selected]
