import ppr
from scipy.sparse import *
from scipy.sparse.linalg import *


def test_trim_vector():
    vector = dok_matrix((5, 1))
    entries = {(0, 0): 5, (1, 0): 7, (2, 0): 4, (3, 0): 8, (4, 0): 3}
    vector.update(entries)
    vector = vector.tocsr()

    trimmed = ppr.trim_vector(vector, 3)
    print(trimmed)
