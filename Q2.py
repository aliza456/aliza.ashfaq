"""
Q2: Return indices of top-k documents by cosine similarity to a query vector.
Cosine similarity = (q·d)/(||q||·||d||).
Break ties by smaller index. Treat similarity as 0 if either vector has zero norm.
Function: top_k_cosine(query, docs, k) -> list[int]
"""

from typing import List
import math

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x*x for x in a))

def top_k_cosine(query: List[float], docs: List[List[float]], k: int) -> List[int]:
    """
    Compute cosine similarities of each doc to query and return indices of top-k.

    Args:
        query: Query vector (length d).
        docs: List of document vectors (each length d).
        k: Number of top indices to return.

    Returns:
        List of indices (0-based) of the top-k documents sorted by descending similarity.
        Ties are broken by smaller index. If k > number of docs, returns as many as available.
        Similarity is 0 if either vector has zero norm.
    """
    n = len(docs)
    k = max(0, min(k, n))

    qn = _norm(query)
    result = []  # list of (sim, idx)
    for i, d in enumerate(docs):
        dn = _norm(d)
        if qn == 0 or dn == 0:
            sim = 0.0
        else:
            sim = _dot(query, d) / (qn * dn)
        result.append((sim, i))

    # Sort by (-similarity, index) to break ties by smaller index
    result.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _, idx in result[:k]]

if __name__ == "__main__":
    q = [1,0]
    docs = [[1,0],[0,1],[1,1],[0,0]]
    print(top_k_cosine(q, docs, 3))  # expect [0,2,1]
