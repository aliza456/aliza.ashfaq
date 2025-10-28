from typing import List

def softmax_classify(W: List[List[float]], b: List[float], X: List[List[float]]) -> List[int]:
    """
    Linear multi-class classification.
    For each x in X, compute logits = W @ x + b and return argmax class index.
    Break ties by choosing the smallest class index.
    """
    if not X:
        return []
    C = len(W)
    preds: List[int] = []
    for x in X:
        best_c = 0
        # compute logit for class 0
        best_val = sum(W[0][j] * x[j] for j in range(len(x))) + (b[0] if b else 0.0)
        for c in range(1, C):
            val = sum(W[c][j] * x[j] for j in range(len(x))) + (b[c] if b else 0.0)
            if val > best_val or (val == best_val and c < best_c):
                best_c = c
                best_val = val
        preds.append(best_c)
    return preds
