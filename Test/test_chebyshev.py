import algorithms.io_utils as io_utils
import algorithms.ppr_interface as ppr_interface
import algorithms.ppr as ppr
import algorithms.vector_utils as vu
import random


def test_chebyshev_equals_standard():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    query_nodes = random.sample(range(weight_matrix.shape[0]), 200)
    alpha = .5

    result = ppr_interface.standard_ppr(weight_matrix, query_nodes, alpha)
    result2 = ppr_interface.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)

    assert(vu.get_l1_distance(result.final_vector, result2.final_vector) < 1E-10)


def test_chebyshev_top_k():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]

    query_nodes = [15, 40, 200]
    alpha = .01
    k = 10

    restart_vector = vu.get_restart_vector(dimension, query_nodes)
    start_vector = restart_vector.copy()

    top_k_result = ppr.chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10)

    result = ppr_interface.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)
    final_vector = vu.trim_vector(result.final_vector, k)

    assert(top_k_result.num_iterations < result.num_iterations)
    assert(vu.get_l1_distance(final_vector, top_k_result.final_vector) < 1E-3)
