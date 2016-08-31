import algorithms.vector_index as vector_index
import algorithms.io_utils as io
import itertools
import random
from scipy.sparse import csr_matrix


def test_build_index():
    weight_matrix = io.load_csr_matrix("Data/test1000.mtx")
    index = vector_index.Vector_Index()
    query_set = random.sample(range(200, 800), 10)
    alphas = [.25, .5]

    index.build_index(weight_matrix, query_set, alphas, num_processes=4)
    assert(all(type(index.get_vector(node_id, alpha)) == csr_matrix for node_id, alpha in itertools.product(*[query_set, alphas])))
