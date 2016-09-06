import time
import algorithms.io_utils as io
import algorithms.vector_utils as vu
import algorithms.ppr as ppr
import algorithms.ppr_interface as ppr_interface


def test_load_youtube():
    start = time.time()
    matrix = io.load_csr_matrix("Data/YouTubeLinkData.mat")
    total_time = time.time() - start
    v = matrix.shape[0]
    e = vu.count_nonzero_entries(matrix)
    print("Loading a matrix with %d vertices and %d edges takes %.2f seconds" % (v, e, total_time))


def test_chopper_top_k_youtube():

    matrix = io.load_csr_matrix("Data/YouTubeLinkData.mat")
    query_nodes = vu.get_query_sets(1, 1, range(matrix.shape[0]))[0]
    alpha = .4
    k = 20

    start = time.time()
    chopper_result = ppr_interface.standard_top_k(matrix, query_nodes, alpha, k)
    chopper_time = time.time() - start
    chopper_iterations = chopper_result.num_iterations
    print("Chopper takes %.3f seconds to process %d iterations with an alpha of %.2f" % (chopper_time, chopper_iterations, alpha))

    start = time.time()
    squeeze_result = ppr_interface.standard_top_k(matrix, query_nodes, alpha, k, top_k_method=ppr.squeeze_top_k)
    squeeze_time = time.time() - start
    squeeze_iterations = squeeze_result.num_iterations

    print("Squeeze takes %.3f seconds to process %d iterations with an alpha of %.2f" % (squeeze_time, squeeze_iterations, alpha))

    print(chopper_result.final_vector)
    print(squeeze_result.final_vector)
