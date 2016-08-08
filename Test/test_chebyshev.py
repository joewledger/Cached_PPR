import io_utils
import ppr
import random
import matplotlib.pyplot as plt
import numpy as np


def test_chebyshev():
    weight_matrix = io_utils.load_csr_matrix("Data/Email-Enron.mat")
    query_nodes = random.sample(range(weight_matrix.shape[0]), 200)
    alpha = .01

    result = ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)
    error_terms = result.error_terms
    print(result.error_terms)

    plt.plot(np.arange(0, len(error_terms), 1), error_terms)
    plt.yscale('log')
    plt.show()
