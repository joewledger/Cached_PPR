import itertools
import random
import io_utils
import ppr
import argparse
import vector_cache


def generate_results(network_filepath, query_sizes, alphas, cache_sizes, methods, num_threads, num_permutations):

    weight_matrix = io_utils.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]
    query_sets = generate_query_sets(num_permutations, max(query_sizes), dimension)
    cache = generate_vector_cache(weight_matrix, query_sets, alphas)
    results = ppr_results(weight_matrix, cache, query_sets, query_sizes, alphas, cache_sizes, methods, num_threads)
    print(results)


def generate_query_sets(num_sets, set_size, query_range):
    return [random.sample(range(query_range), set_size) for _ in range(num_sets)]


def generate_vector_cache(weight_matrix, query_sets, alphas):
    all_query_nodes = set(itertools.chain(*query_sets))
    cache = vector_cache.vector_cache()
    for query_node, alpha in itertools.product(all_query_nodes, alphas):
        vector = ppr.get_proximity_vector(weight_matrix, query_node, alpha)
        cache.insert_vector(query_node, alpha, vector)
    return cache


def ppr_results(weight_matrix, cache, query_sets, query_sizes, alphas, cache_sizes, methods, num_threads):
    for query_set in query_sets:
        for query_size, alpha, cache_size in itertools.product(*[query_sizes, alphas, cache_sizes]):


def plot_results():
    return None


def cached_ppr_results(weight_matrix, cached_vectors, alpha, cache_size, norm_method):
    methods = {1: None, 2: "total_sum", 3: "twice_normalized"}
    return ppr.cached_ppr(weight_matrix, cached_vectors, alpha, norm_method=methods[norm_method])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generates results for cached PPR experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--query_sizes', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--cache_sizes', type=int, nargs='+')
    parser.add_argument('--methods', type=str, nargs='+')
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--num_permutations', type=int)
    parser.set_defaults(network_filepath="Data/Email-Enron.mat", query_sizes=[10, 50, 200],
                        alphas=[.01, .1, .25], cache_sizes=[10, 100, 1000],
                        methods=[x for x in range(4)], num_threads=5, num_permutations=10)

    args = parser.parse_args()

    generate_results(args.network_filepath, args.query_sizes, args.alphas,
                     args.cache_sizes, args.methods, args.num_threads, args.num_permutations)
