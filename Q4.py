"""
Q4: Single epoch of binary perceptron.
Labels are -1 or +1. For each sample (in order), if y_i*(w·x + b) <= 0, update:
    w = w + lr*y_i*x
    b = b + lr*y_i
Return [w, b].

Function: perceptron_epoch(X, y, w, b, lr) -> [list[float], float]
"""

from typing import List, Tuple

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def perceptron_epoch(
    X: List[List[float]],
    y: List[int],
    w: List[float],
    b: float,
    lr: float
) -> Tuple[List[float], float]:
    """
    Run a single perceptron epoch over the dataset in-order.

    Args:
        X: List of n samples; each is a list of floats (features).
        y: List of n labels; each label is -1 or +1.
        w: Current weight vector.
        b: Current bias (float).
        lr: Learning rate (float).

    Returns:
        (w, b) after one epoch.
    """
    w = list(w)  # copy to avoid in-place mutation surprises
    for xi, yi in zip(X, y):
        activation = _dot(w, xi) + b
        if yi * activation <= 0:
            # w = w + lr * yi * xi
            for j in range(len(w)):
                w[j] += lr * yi * xi[j]
            b += lr * yi
    return w, b

if __name__ == "__main__":
    X = [[1,1], [2,2], [-1,-1]]
    y = [1, 1, -1]
    w, b = [0.0, 0.0], 0.0
    print(perceptron_epoch(X, y, w, b, 1.0))
