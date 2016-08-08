import itertools
import io_utils as io
import ppr
import argparse
import vector_cache
import vector_utils as vu


class Param_Object:

    def __init__(self, weight_matrix, cache, query_set, query_size, alpha, cache_size, num_permutations):
        self.weight_matrix = weight_matrix
        self.cache = cache
        self.query_set = query_set
        self.query_size = query_size
        self.alpha = alpha
        self.cache_size = cache_size
        self.num_permutations = num_permutations


def generate_results(network_filepath, save_file, query_sizes, alphas, cache_sizes, num_permutations):

    query_node_filepath = network_filepath.replace("Data/", "Cache/")[:-4] + "-queries.pickle"
    cache_filepath = network_filepath.replace("Data/", "Cache/")[:-4] + ".pickle"

    weight_matrix = io.load_csr_matrix(network_filepath)
    query_nodes = io.load_query_nodes(query_node_filepath)
    query_sets = vu.get_query_sets(num_permutations, max(query_sizes), query_nodes)
    cache = vector_cache.vector_cache()
    cache.load_from_file(cache_filepath)
    param_objects = [Param_Object(weight_matrix, cache, q_set, q_size, a, c_size, num_permutations) for q_set, q_size,
                     a, c_size in itertools.product(*[query_sets, query_sizes, alphas, cache_sizes])]
    result = [get_ppr_results(p) for p in param_objects]
    flatten = lambda l: [item for sublist in l for item in sublist]
    result = flatten(result)
    writer = open(save_file, "w")
    for r in result:
        writer.write("\t".join(str(x) for x in r) + "\n")
    writer.close()


def get_ppr_results(param_object):
    query_size = param_object.query_size
    weight_matrix = param_object.weight_matrix
    alpha = param_object.alpha
    cache = param_object.cache
    cache_size = param_object.cache_size

    q = param_object.query_set[:query_size]

    results = {}
    results["standard"] = ppr.standard_ppr(weight_matrix, q, alpha)
    results["total_sum"] = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, norm_method=vu.total_sum)
    results["twice_normalized"] = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, norm_method=vu.twice_normalized)
    results["chebyshev_standard"] = ppr.standard_ppr(weight_matrix, q, alpha, ppr_method=ppr.chebyshev_ppr)
    results["chebyshev_total_sum"] = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.total_sum)
    results["chebyshev_twice_normalized"] = ppr.cached_ppr(weight_matrix, q, cache, cache_size, alpha, ppr_method=ppr.chebyshev_ppr, norm_method=vu.twice_normalized)
    return [(key, query_size, alpha, cache_size, value.num_iterations) for key, value in results.items()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generates results for cached PPR experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--query_sizes', type=int, nargs='+')
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--cache_sizes', type=int, nargs='+')
    parser.add_argument('--num_permutations', type=int)

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", save_file="Results/out.txt", query_sizes=[10, 50, 200],
                        alphas=[.01, .1, .25], cache_sizes=[10, 100, 1000], num_permutations=10)

    args = parser.parse_args()

    generate_results(args.network_filepath, args.save_file, args.query_sizes, args.alphas,
                     args.cache_sizes, args.num_permutations)
