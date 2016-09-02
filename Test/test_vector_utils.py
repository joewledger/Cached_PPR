import algorithms.io_utils as io
import algorithms.vector_utils as vu
import algorithms.vector_index as vector_index
from scipy.sparse import *


def test_start_vectors():
    weight_matrix = io.load_csr_matrix("Data/test.mtx")
    query_sets = [[5, 37, 84]]
    alphas = [.01]
    index_size = 10

    index = vector_index.Vector_Index()
    index.build_index(weight_matrix, query_sets, alphas)
    vector_list = index.get_vector_list(query_sets[0], alphas[0], index_size=index_size)

    u = vu.unnormalized(vector_list)
    ts = vu.total_sum(vector_list)
    tn = vu.twice_normalized(vector_list)

    assert(abs(ts.sum() - tn.sum()) < 1E-5)
    assert(abs(ts.sum() - u.sum()) > index_size * alphas[0])


def test_non_zero_indices_set():
    dimension = 100
    vector = dok_matrix((dimension, 1))
    vector.update({(i, 0): (1.0 if i % 2 == 0 else 2.0) for i in range(dimension)})
    vector = vector.tocsr()

    vector = vu.trim_vector(vector, dimension / 2)
    nonzeroes = vu.get_nonzero_indices_set(vector)
    assert(nonzeroes == {i for i in range(dimension) if i % 2 == 1})
