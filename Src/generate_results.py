import itertools
import io_utils
import ppr
import argparse
import vector_cache
import vector_utils as vu


def generate_results(network_filepath, save_file, query_sizes, alphas, cache_sizes, num_threads, num_permutations):

    weight_matrix = io_utils.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]
    query_sets = vu.get_query_sets(num_permutations, max(query_sizes), dimension)
    cache = get_vector_cache(weight_matrix, query_sets, alphas)
    save_results(save_file, weight_matrix, cache, query_sets, query_sizes, alphas, cache_sizes, num_threads)


def get_vector_cache(weight_matrix, query_sets, alphas):
    cache = vector_cache.vector_cache()
    cache.build_cache(weight_matrix, query_sets, alphas)
    return cache


def save_results(save_file, weight_matrix, cache, query_sets, query_sizes, alphas, cache_sizes, num_threads):
    writer = open(save_file, "w")
    for query_set, query_size, alpha, cache_size in itertools.product(*[query_sets, query_sizes, alphas, cache_sizes]):
        q = query_set[:query_size]
        standard = ppr.standard_ppr(weight_matrix, q, alpha).num_iterations
        total_sum = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, norm_method=vu.total_sum).num_iterations
        twice_normalized = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, norm_method=vu.twice_normalized).num_iterations
        chebyshev_standard = ppr.standard_ppr(weight_matrix, q, alpha, ppr_method=ppr.chebyshev_ppr).num_iterations
        chebyshev_total_sum = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.total_sum).num_iterations
        chebyshev_twice_normalized = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.twice_normalized).num_iterations
        writer.write("\t".join(str(x) for x in ["standard", query_size, alpha, cache_size, standard]) + "\n")
        writer.write("\t".join(str(x) for x in ["total_sum", query_size, alpha, cache_size, total_sum]) + "\n")
        writer.write("\t".join(str(x) for x in ["twice_normalized", query_size, alpha, cache_size, twice_normalized]) + "\n")
        writer.write("\t".join(str(x) for x in ["chebyshev_standard", query_size, alpha, cache_size, chebyshev_standard]) + "\n")
        writer.write("\t".join(str(x) for x in ["chebyshev_total_sum", query_size, alpha, cache_size, chebyshev_total_sum]) + "\n")
        writer.write("\t".join(str(x) for x in ["chebyshev_twice_normalized", query_size, alpha, cache_size, chebyshev_twice_normalized]) + "\n")
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generates results for cached PPR experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--query_sizes', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--cache_sizes', type=int, nargs='+')
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--num_permutations', type=int)
    parser.add_argument('--save_file', type=str)
    parser.set_defaults(network_filepath="Data/Email-Enron.mat", query_sizes=[10, 50, 200],
                        alphas=[.01, .1, .25], cache_sizes=[10, 100, 1000],
                        num_threads=5, num_permutations=10, save_file="Results/out.txt")

    args = parser.parse_args()

    generate_results(args.network_filepath, args.save_file, args.query_sizes, args.alphas,
                     args.cache_sizes, args.num_threads, args.num_permutations)
