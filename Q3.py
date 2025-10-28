"""
Q3: Standardize features column-wise (z-score with population std, ddof=0).
For each column j: z = (x - mean_j)/std_j. If std_j == 0, return zeros for that column.
Round each value to 4 decimals.
Function: standardize_columns(X) -> list[list[float]]
"""

from typing import List
import math

def standardize_columns(X: List[List[float]]) -> List[List[float]]:
    """
    Standardize columns with population std (ddof=0).

    Args:
        X: Matrix of size n x d (n rows, d columns).

    Returns:
        Standardized matrix of same shape. Each value rounded to 4 decimals.
        Columns with zero std become all zeros.
    """
    if not X:
        return []
    n = len(X)
    d = len(X[0]) if X[0] else 0
    if d == 0:
        return [[] for _ in range(n)]

    # Compute column means
    means = [0.0]*d
    for row in X:
        for j, val in enumerate(row):
            means[j] += val
    means = [m / n for m in means]

    # Compute population std (sqrt( mean( (x-mean)^2 ) ))
    variances = [0.0]*d
    for row in X:
        for j, val in enumerate(row):
            variances[j] += (val - means[j])**2
    variances = [v / n for v in variances]
    stds = [math.sqrt(v) for v in variances]

    # Standardize
    Z = [[0.0]*d for _ in range(n)]
    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if stds[j] == 0:
                z = 0.0
            else:
                z = (val - means[j]) / stds[j]
            Z[i][j] = round(z, 4)
    return Z

if __name__ == "__main__":
    X = [[1, 2, 3],
         [2, 2, 5],
         [3, 2, 7]]
    print(standardize_columns(X))  # column 2 std=0 → zeros there
