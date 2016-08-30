import argparse
import io_utils as io
import numpy as np
import vector_cache
import ppr
import vector_utils as vu
from collections import OrderedDict
import matplotlib.pyplot as plt


def plot_cache_size(network_filepath, output_file, query_size, alpha, network_size_divisor):
    cache_filepath = io.get_cache_filepath(network_filepath)

    weight_matrix = io.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]
    dividand = int(dimension / network_size_divisor)
    ind = list(np.arange(dividand, dimension, dividand))
    ind.append(dimension)

    cache = vector_cache.vector_cache()
    cache.load_from_file(cache_filepath)

    query_nodes = vu.get_query_sets(1, query_size, range(dimension))[0]

    results = OrderedDict()
    results["Standard"] = [ppr.standard_ppr(weight_matrix, query_nodes, alpha).num_iterations] * len(ind)
    results["Total Sum"] = [ppr.cached_ppr(weight_matrix, query_nodes, cache, c_size, alpha, norm_method=vu.total_sum).num_iterations for c_size in ind]
    results["Twice Normalized"] = [ppr.cached_ppr(weight_matrix, query_nodes, cache, c_size, alpha, norm_method=vu.twice_normalized).num_iterations for c_size in ind]
    results["Chebyshev Standard"] = [ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr).num_iterations] * len(ind)
    results["Chebyshev Total Sum"] = [ppr.cached_ppr(weight_matrix, query_nodes, cache, c_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.total_sum).num_iterations for c_size in ind]
    results["Chebyshev Twice Normalized"] = [ppr.cached_ppr(weight_matrix, query_nodes, cache, c_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.twice_normalized).num_iterations for c_size in ind]

    for key, value in results.items():
        plt.plot(ind, value, label=key)
    plt.title(r'Cache Size vs. Number of Iterations ($\alpha$=%.2f, |Q|=%d)' % (alpha, query_size))
    plt.xlabel("Cache Size")
    plt.ylabel("Number of Iterations")
    plt.ylim([0, max(value[0] for value in results.values()) * 2])
    plt.legend()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plots Cache Size vs. Number of Iterations for PPR experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--query_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--network_size_divisor', type=int)
    parser.set_defaults(network_filepath="Data/Email-Enron.mat", output_file="Plots/cache_size_plot.png", query_size=200,
                        alpha=.01, network_size_divisor=100)

    args = parser.parse_args()
    plot_cache_size(args.network_filepath, args.output_file, args.query_size, args.alpha, args.network_size_divisor)
