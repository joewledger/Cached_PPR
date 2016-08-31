import algorithms.io_utils as io
import algorithms.vector_utils as vu
import algorithms.vector_index as vector_index


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
