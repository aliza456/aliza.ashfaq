from typing import List

def knn_predict(train_X: List[List[float]], train_y: List[int], test_X: List[List[float]], k: int) -> List[int]:
    """
    Predict labels for test_X using k-NN with Euclidean distance.
    Neighbors sorted by (distance, label); vote by majority; tie-break by smallest label.
    """
    if not test_X:
        return []
    n_train = len(train_X)
    if n_train == 0:
        return [0 for _ in test_X]  # undefined; return 0s to avoid crash
    k = max(1, min(k, n_train))
    preds: List[int] = []
    for x in test_X:
        # compute squared distances (monotonic with Euclidean)
        dists = []
        for i, t in enumerate(train_X):
            dist = 0.0
            for a, b in zip(x, t):
                diff = a - b
                dist += diff * diff
            dists.append((dist, train_y[i]))
        dists.sort(key=lambda t: (t[0], t[1]))
        # take top-k and vote
        counts = {}
        for _, lbl in dists[:k]:
            counts[lbl] = counts.get(lbl, 0) + 1
        # choose label with max count; tie -> smallest label
        best_label = None
        best_count = -1
        for lbl in sorted(counts.keys()):  # sorted ensures smaller label wins ties
            c = counts[lbl]
            if c > best_count:
                best_count = c
                best_label = lbl
        preds.append(int(best_label))
    return preds
