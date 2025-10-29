from typing import List

def kmeans_assign(points: List[List[float]], centroids: List[List[float]]) -> List[int]:
    """
    For each point, return the index (0-based) of the nearest centroid by Euclidean distance.
    Break ties by choosing the smaller centroid index.
    """
    if not centroids:
        return [-1 for _ in points]
    out: List[int] = []
    for p in points:
        best_idx = 0
        best_d = sum((pi - centroids[0][i])**2 for i, pi in enumerate(p))
        for j in range(1, len(centroids)):
            d = sum((pi - centroids[j][i])**2 for i, pi in enumerate(p))
            if d < best_d or (d == best_d and j < best_idx):
                best_idx, best_d = j, d
        out.append(best_idx)
    return out
