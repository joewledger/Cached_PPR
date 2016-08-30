import io_utils
import ppr
import random
import vector_utils as vu


def test_chebyshev_equals_standard():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    query_nodes = random.sample(range(weight_matrix.shape[0]), 200)
    alpha = .5

    result = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
    result2 = ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)

    assert(ppr.get_l1_norm(result.final_vector, result2.final_vector) < 1E-10)


def test_chebyshev_top_k():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    
    query_nodes = [15, 40, 200]
    alpha = .01
    k = 10

    restart_vector = ppr.get_restart_vector(dimension, query_nodes)
    start_vector = restart_vector.copy()


    top_k_result = ppr.chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10)

    result = ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)
    final_vector = ppr.trim_vector(result.final_vector, k)

    assert(top_k_result.num_iterations < result.num_iterations)
    assert(ppr.get_l1_norm(final_vector, top_k_result.final_vector) < 1E-3)


def test_indexed_chebyshev_top_k():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    query_nodes = random.sample(range(dimension),2)
    alpha = .01
    k = 10

    restart_vector = ppr.get_restart_vector(dimension, query_nodes)
    start_vector = restart_vector.copy()
    top_k_result = ppr.chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10)
    print(top_k_result.final_vector)
    print(top_k_result.num_iterations)


    vector_collection = [ppr.get_proximity_vector(weight_matrix, q, alpha) for q in query_nodes]
    start_vector = vu.twice_normalized(vector_collection)

    indexed_top_k_result = ppr.indexed_chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10)
    print(indexed_top_k_result.final_vector)
    print(indexed_top_k_result.num_iterations)

    print(ppr.get_l1_norm(indexed_top_k_result.final_vector, top_k_result.final_vector))
