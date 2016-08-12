import io_utils
import ppr
import random


def test_chebyshev():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    query_nodes = random.sample(range(weight_matrix.shape[0]), 200)
    alpha = .5

    result = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
    result2 = ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)

    print(ppr.trim_vector(result.final_vector, 10))
    print("\n")
    print(ppr.trim_vector(result2.final_vector, 10))
