import ppr
import io_utils as io
import vector_utils as vu
import vector_cache as vc
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import OrderedDict


def plot(output_dir, weight_matrix, query_set, vector_cache, alpha, query_size, cache_size):
    standard = lambda ppr_method: ppr.standard_ppr(weight_matrix, query_set, alpha, ppr_method=ppr_method)
    cached = lambda ppr_method, norm_method: ppr.cached_ppr(weight_matrix, query_set, vector_cache,
                                                            cache_size, alpha, ppr_method=ppr_method,
                                                            norm_method=norm_method)
    results = OrderedDict([("Standard", standard(ppr.ppr)),
                           ("Twice Normalized", cached(ppr.ppr, vu.twice_normalized)),
                           ("Total Sum", cached(ppr.ppr, vu.total_sum)),
                           ("Chebyshev Standard", standard(ppr.chebyshev_ppr)),
                           ("Chebyshev Twice Normalized", cached(ppr.chebyshev_ppr, vu.twice_normalized)),
                           ("Chebyshev Total Sum", cached(ppr.chebyshev_ppr, vu.total_sum))])

    dict_subset = lambda original, keys: OrderedDict([(key, original[key]) for key in keys])
    output_file = lambda s: "%serror_%s_alpha_%.2f_q_%d_cache_size_%d.png" % (output_dir, s, alpha, query_size, cache_size)

    plotting_subsets = {output_file("standard"): dict_subset(results, ["Standard", "Twice Normalized", "Total Sum"]),
                        output_file("chebyshev"): dict_subset(results, ["Chebyshev Standard", "Chebyshev Twice Normalized", "Chebyshev Total Sum"]),
                        output_file("both"): results}

    for outfile, result in plotting_subsets.items():
        plot_subset(outfile, result)


def plot_subset(output_file, results):
    for label, result in results.items():
        e = result.error_terms
        plt.plot(np.arange(0, len(e), 1), e, label=label)
    plt.ylabel("Size of Error")
    plt.xlabel("Number of Iterations")
    plt.yscale('log')
    plt.legend()
    plt.title("Error Terms vs. Number of Iterations")
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    input_file = "Data/Email-Enron.mat"
    output_dir = "Plots/Error_Iterations/"

    alphas = [.01, .1, .25]
    query_sizes = [10, 50, 200]
    cache_sizes = [10, 100, 1000]

    weight_matrix = io.load_csr_matrix(input_file)

    query_set = vu.get_query_sets(1, 200, weight_matrix.shape[0])[0]
    cache = vc.vector_cache()
    cache.build_cache(weight_matrix, query_set, alphas)

    for alpha, query_size, cache_size in itertools.product(*[alphas, query_sizes, cache_sizes]):
        plot(output_dir, weight_matrix, query_set, cache, alpha, query_size, cache_size)
        print(alpha, query_size, cache_size)
