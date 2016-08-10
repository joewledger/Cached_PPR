import matplotlib.pyplot as plt
import io_utils as io
import vector_cache
import vector_utils as vu
import itertools
import argparse
import ppr
import numpy as np


def plot_all(network_filepath, save_dir, query_sizes, alphas, cache_sizes, num_permutations):

    weight_matrix = io.load_csr_matrix(network_filepath)
    cache = vector_cache.vector_cache()
    print("Started loading cache")
    cache.load_from_file(io.get_cache_filepath(network_filepath))
    print("Done loading cache")
    query_nodes = io.load_query_nodes(io.get_cached_queries_filepath(network_filepath))
    query_sets = vu.get_query_sets(num_permutations, max(query_sizes), query_nodes)
    for query_size, cache_size in itertools.product(*[query_sizes, cache_sizes]):
        plot_incorrect_alpha(save_dir, weight_matrix, query_sets, cache, alphas, query_size, cache_size)


def plot_incorrect_alpha(save_dir, weight_matrix, query_sets, cache, alphas, query_size, cache_size):

    grid = get_comparison_grid(weight_matrix, query_sets, cache, alphas, query_size, cache_size)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    img = axes.imshow(grid, interpolation='none', cmap='bwr_r')
    axes.set_title(r"Incorrect $\alpha$ relative performance (|Q|=%d, |I|=%d)" % (query_size, cache_size))
    axes.set_xlabel(r"$\alpha$")
    axes.set_ylabel(r"Indexed $\alpha$")
    ticks = np.arange(0, len(alphas), 1)
    tick_labels = ["%.2f" % f for f in alphas]
    axes.set_xticks(ticks)
    axes.set_yticks(ticks)
    axes.set_xticklabels(tick_labels)
    axes.set_yticklabels(reversed(tick_labels))
    fig.colorbar(img)
    save_file = "%sincorrect_alpha_plot_q_%d_c_%d.png" % (save_dir, query_size, cache_size)
    plt.savefig(save_file)


def float_key(f):
    return "%.3f" % f


def get_comparison_grid(weight_matrix, query_sets, cache, alphas, query_size, cache_size):

    baseline, comparison = get_baseline_and_comparison(weight_matrix, query_sets, cache, alphas, query_size, cache_size)
    comparison_dict = get_comparision_dict(baseline, comparison)
    grid = []
    for ca in reversed(alphas):
        grid.append([comparison_dict[(float_key(aa), float_key(ca))] for aa in alphas])
    return grid


def get_comparision_dict(baseline, comparison):
    comparison_dict = {}
    for alpha1, alpha2 in comparison.keys():
        c = np.array(comparison[(alpha1, alpha2)])
        b = np.array(baseline[alpha1])
        comparison_dict[(alpha1, alpha2)] = np.mean(c) / np.mean(b)
    return comparison_dict


def get_baseline_and_comparison(weight_matrix, query_sets, cache, alphas, query_size, cache_size):
    baseline = {float_key(alpha): [] for alpha in alphas}
    comparison = {(float_key(a1), float_key(a2)): [] for a1, a2 in itertools.product(*[alphas, alphas])}

    for query_set in query_sets:
        for actual_alpha in alphas:
            baseline_iterations = ppr.standard_ppr(weight_matrix, query_set, actual_alpha, ppr_method=ppr.chebyshev_ppr).num_iterations
            baseline[float_key(actual_alpha)].append(baseline_iterations)
            for cache_alpha in alphas:
                comparison_iterations = incorrect_alpha_ppr(weight_matrix, query_set, actual_alpha, cache_alpha,
                                                            cache, cache_size, ppr.chebyshev_ppr, vu.twice_normalized).num_iterations
                comparison[(float_key(actual_alpha), float_key(cache_alpha))].append(comparison_iterations)
    return baseline, comparison


def incorrect_alpha_ppr(weight_matrix, query_nodes, alpha, cache_alpha, cache, cache_size, ppr_method, norm_method):
    dimension = weight_matrix.shape[0]

    vector_list = cache.get_vector_list(query_nodes, cache_alpha, cache_size=cache_size)
    start_vector = norm_method(vector_list)
    restart_vector = ppr.get_restart_vector(dimension, query_nodes)
    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Plots effect of using incorrect value for alpha")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--query_sizes', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--cache_sizes', type=int, nargs='+')
    parser.add_argument('--num_permutations', type=int)

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", save_dir="Plots/Incorrect_Alphas/", query_sizes=[10, 50, 200],
                        alphas=[.01, .1, .25], cache_sizes=[10, 100, 1000], num_permutations=10)

    args = parser.parse_args()

    plot_all(args.network_filepath, args.save_dir, args.query_sizes, args.alphas, args.cache_sizes, args.num_permutations)
