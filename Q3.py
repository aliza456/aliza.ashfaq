from typing import List

def minmax_scale(X: List[List[float]]) -> List[List[float]]:
    """
    Min-max scale each column to [0,1].
    For column j: (x - min_j) / (max_j - min_j).
    If max_j == min_j, return zeros for that column.
    Round each value to 4 decimals.
    """
    if not X:
        return []
    n = len(X)
    d = len(X[0]) if X[0] else 0
    if d == 0:
        return [[] for _ in range(n)]
    mins = [float('inf')] * d
    maxs = [float('-inf')] * d
    for row in X:
        for j, v in enumerate(row):
            if v < mins[j]: mins[j] = v
            if v > maxs[j]: maxs[j] = v
    Y = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            denom = maxs[j] - mins[j]
            y = 0.0 if denom == 0 else (X[i][j] - mins[j]) / denom
            Y[i][j] = float(f"{y:.4f}")
    return Y
