"""
Q1: Implement k-means assignment step.
Function: kmeans_assign(points, centroids) -> list[int]

For each point, return the index (0-based) of the nearest centroid by Euclidean distance.
Break ties by choosing the smaller centroid index.
"""

from typing import List

def _squared_euclidean(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))

def kmeans_assign(points: List[List[float]], centroids: List[List[float]]) -> List[int]:
    """
    Assign each point to the nearest centroid using Euclidean distance.

    Args:
        points: List of n points, each a list of floats of dimension d.
        centroids: List of m centroids, each a list of floats of dimension d.

    Returns:
        A list of length n where each entry is the index (0-based) of the nearest centroid.
        Ties are broken by choosing the smaller centroid index.
    """
    if not centroids:
        return [ -1 for _ in points ]  # no centroids; return -1 as a sentinel

    assignments: List[int] = []
    for p in points:
        best_idx = 0
        best_dist = _squared_euclidean(p, centroids[0])
        for j in range(1, len(centroids)):
            d = _squared_euclidean(p, centroids[j])
            # Tie-breaker: smaller index wins
            if d < best_dist or (d == best_dist and j < best_idx):
                best_idx, best_dist = j, d
        assignments.append(best_idx)
    return assignments

if __name__ == "__main__":
    # Simple sanity check
    pts = [[0,0],[2,2],[10,10]]
    cents = [[0,0],[5,5]]
    print(kmeans_assign(pts, cents))  # [0,0,1]
