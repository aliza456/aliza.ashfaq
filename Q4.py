from typing import List

def bow_transform(corpus: List[str], vocab: List[str]) -> List[List[int]]:
    """
    Tokenize by whitespace (lowercase assumed), mark vocab terms per document (binary presence),
    and return a 2D list with shape [len(corpus)][len(vocab)].
    Terms not in vocab are ignored.
    """
    m = len(corpus)
    n = len(vocab)
    if m == 0:
        return []
    index = {term: j for j, term in enumerate(vocab)}
    mat = [[0] * n for _ in range(m)]
    for i, doc in enumerate(corpus):
        seen = set()
        for tok in doc.split():
            j = index.get(tok)
            if j is not None:
                seen.add(j)
        for j in seen:
            mat[i][j] = 1
    return mat
