from typing import List

def standardize_columns(X: List[List[float]]) -> List[List[float]]:
    """
    Standardize features column-wise using population std (ddof=0).
    For each column j: z = (x - mean_j) / std_j.
    If std_j == 0, the entire column becomes zeros.
    Each value rounded to 4 decimals.
    """
    if not X:
        return []
    n = len(X)
    d = len(X[0]) if X[0] else 0
    if d == 0:
        return [[] for _ in range(n)]
    means = [0.0] * d
    for row in X:
        for j, v in enumerate(row):
            means[j] += v
    means = [m / n for m in means]
    variances = [0.0] * d
    for row in X:
        for j, v in enumerate(row):
            dv = v - means[j]
            variances[j] += dv * dv
    variances = [v / n for v in variances]
    stds = [v ** 0.5 for v in variances]
    Z = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            z = 0.0 if stds[j] == 0 else (X[i][j] - means[j]) / stds[j]
            Z[i][j] = float(f"{z:.4f}")
    return Z
