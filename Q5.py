from typing import List, Dict

def confusion_matrix(y_true: List, y_pred: List, labels: List) -> List[List[int]]:
    """
    Compute confusion matrix: rows=true labels, cols=predicted labels, order by `labels`.
    """
    n = len(labels)
    idx: Dict = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    return mat
