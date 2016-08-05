import ppr
import io_utils as io
import random
import start_vectors as sv
import vector_cache as vc
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict


def plot_error_iterations(input_file, output_file):

    weight_matrix = io.load_csr_matrix(input_file)
    alpha = .01
    query_size = 10
    cache_size = 100
    query_set = random.sample(range(weight_matrix.shape[0]), query_size)

    vector_cache = vc.vector_cache()
    vector_cache.build_cache(weight_matrix, [query_set], [alpha])

    result = ppr.standard_ppr(weight_matrix, query_set, alpha)
    result2 = ppr.cached_ppr(weight_matrix, query_set, vector_cache, cache_size, alpha, norm_method=sv.twice_normalized)
    result3 = ppr.cached_ppr(weight_matrix, query_set, vector_cache, cache_size, alpha, norm_method=sv.total_sum)

    result_labels = OrderedDict([("Standard", result), ("Twice Normalized", result2), ("Total Sum", result3)])

    for label, result in result_labels.items():
        e = result.error_terms
        plt.plot(np.arange(0, len(e), 1), e, label=label)
    plt.ylabel("Size of Error")
    plt.xlabel("Number of Iterations")
    plt.yscale('log')
    plt.legend()
    plt.title("Error Terms vs. Number of Iterations")
    plt.show()

if __name__ == "__main__":
    #input_file = "Data/Email-Enron.mat"
    input_file = "Data/test1000.mtx"
    output_file = "Results/plot_error_iterations.png"

    plot_error_iterations(input_file, output_file)
