"""
Q5: Compute a confusion matrix.
Rows correspond to true labels and columns to predicted labels, in the given order.
Function: confusion_matrix(y_true, y_pred, labels) -> list[list[int]]
"""

from typing import List, Dict

def confusion_matrix(y_true: List, y_pred: List, labels: List) -> List[List[int]]:
    """
    Build confusion matrix with rows=true labels and columns=predicted labels.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels (same length as y_true).
        labels: List of all labels in the desired order for rows and columns.

    Returns:
        2D list of shape (len(labels), len(labels)) with integer counts.
    """
    n = len(labels)
    idx: Dict = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:  # silently ignore unknown labels
            mat[idx[t]][idx[p]] += 1
    return mat

if __name__ == "__main__":
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 2, 2, 1, 0]
    labels = [0, 1, 2]
    for row in confusion_matrix(y_true, y_pred, labels):
        print(row)
