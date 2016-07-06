from scipy.sparse.linalg import *
from scipy.sparse import *
import random
import ppr

def test_limit_top_k():
    size = 40000
    vector = dok_matrix((size, 1))
    entries = {(x, 0) : random.random() for x in range(size)}
    vector.update(entries)
    vector = ppr.limit_vector_top_k(vector, 10)
    print(vector)
