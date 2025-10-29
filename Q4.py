from typing import List

def bow_transform(corpus: List[str], vocab: List[str]) -> List[List[int]]:
    """
    Tokenize by whitespace (lowercase assumed), count vocab terms per document,
    and return a 2D list with shape [len(corpus)][len(vocab)].
    Terms not in vocab are ignored.
    """
    m = len(corpus)
    n = len(vocab)
    if m == 0:
        return []
    index = {term: j for j, term in enumerate(vocab)}
    mat = [[0]*n for _ in range(m)]
    for i, doc in enumerate(corpus):
        for tok in doc.split():
            j = index.get(tok)
            if j is not None:
                mat[i][j] += 1
    return mat
