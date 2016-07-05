import itertools
import random
import io_utils
import ppr
import sqlite3


def generate_query_sets(query_size, query_range, num_permutations):
    return [random.sample(range(0, query_range), query_size) for n in range(0, num_permutations)]


def generate_parameter_combinations():
    query_sizes = [10, 50, 200]
    alphas = [.01, .1, .25]
    cache_sizes = [10, 100, 1000]
    norm_methods = ["total_sum", "twice_normalized", None]

    all_params = [query_sizes, alphas, cache_sizes, norm_methods]
    return [x for x in itertools.product(*all_params)]


def standard_ppr_parameter_combinations():
    query_sizes = [10, 50, 200]
    alphas = [.01, .1, .25]
    params = [query_sizes, alphas]
    return [x for x in itertools.product(*params)]


def generate_cached_ppr_results(matrix_file, weight_matrix, query_sets):

    parameters = generate_parameter_combinations()

    for p in parameters:
        query_size, alpha, cache_size, norm_method = p

        for run_id, q in enumerate(query_sets):
            query_nodes = q[:query_size]
            cached_vectors = io_utils.load_cached_vectors(db_file, query_nodes, alpha, cache_size)
            v, i, e = ppr.cached_ppr(weight_matrix, cached_vectors, alpha, norm_method=norm_method)
            yield [matrix_file, run_id, query_size, alpha, cache_size, norm_method, i, 0]


def generate_standard_ppr_results(matrix_file, weight_matrix, query_sets):
    parameters = standard_ppr_parameter_combinations()

    for p in parameters:
        query_size, alpha = p

        for run_id, q in enumerate(query_sets):
            query_nodes = q[:query_size]
            v, i, e = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
            yield [matrix_file, run_id, query_size, alpha, -1, "Standard", i, 0]

if __name__ == '__main__':

    num_permutations = 10

    matrix_file = "Data/Email-Enron.mat"
    db_file = "Cache/proximity_vectors.sqlite3"
    weight_matrix = io_utils.read_csr_matrix(matrix_file)

    query_sets = generate_query_sets(200, weight_matrix.shape[0], 1)

    cached_results = generate_cached_ppr_results(matrix_file, weight_matrix, query_sets)
    io_utils.save_results(db_file, cached_results)

    standard_results = generate_standard_ppr_results(matrix_file, weight_matrix, query_sets)
    io_utils.save_results(db_file, standard_results)
