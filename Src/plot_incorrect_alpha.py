import matplotlib.pyplot as plt
import io_utils as io
import vector_cache
import vector_utils as vu
import itertools
import argparse
import ppr


def plot_all(network_filepath, save_dir, query_sizes, alphas, cache_sizes):
    weight_matrix = io.load_csr_matrix(network_filepath)
    cache = vector_cache.vector_cache()
    print("Started loading cache")
    cache.load_from_file(io.get_cache_filepath(network_filepath))
    print("Done loading cache")
    query_nodes = io.load_query_nodes(io.get_cached_queries_filepath(network_filepath))
    query_set = vu.get_query_sets(1, 200, query_nodes)[0]
    for query_size, cache_size in itertools.product(*[query_sizes, cache_sizes]):
        plot_incorrect_alpha(save_dir, weight_matrix, query_set[:query_size], cache, alphas, query_size, cache_size)


def plot_incorrect_alpha(save_dir, weight_matrix, query_set, cache, alphas, query_size, cache_size):
    save_file = "%s_incorrect_alpha_plot_q_%d_c_%d.png" % (save_dir, query_size, cache_size)
    result = ppr.standard_ppr(weight_matrix, query_set, alphas[0], ppr_method=ppr.chebyshev_ppr)
    result2 = ppr.cached_ppr(weight_matrix, query_set, cache, cache_size, alphas[0], norm_method=vu.twice_normalized, ppr_method=ppr.chebyshev_ppr)

    print(result.num_iterations, result2.num_iterations)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plots effect of using incorrect value for alpha")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--query_sizes', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--cache_sizes', type=int, nargs='+')

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", save_dir="Plots/Incorrect_Alphas/", query_sizes=[10, 50, 200],
                        alphas=[.01, .1, .25], cache_sizes=[10, 100, 1000])

    args = parser.parse_args()

    plot_all(args.network_filepath, args.save_dir, args.query_sizes, args.alphas, args.cache_sizes)
