import generate_results as gr
import itertools
import matplotlib as plt
import pickle
import os.path
import io_utils
import random
import ppr

db_file = "Cache/proximity_vectors.sqlite3"
base_out = "Results/"

query_sizes = [10, 50, 200]
cache_sizes = [10, 100, 1000]
alphas = [.01, .1, .25]
norm_methods = ["total_sum", "twice_normalized"]

cached_params = [x for x in itertools.product(*[query_sizes, cache_sizes, alphas, norm_methods])][:3]
standard_params = [x for x in itertools.product(*[query_sizes, alphas])][:3]


def janky_results(filename):
    db_file = "Cache/proximity_vectors.sqlite3"

    weight_matrix = io_utils.read_csr_matrix(filename)
    query_nodes = random.sample(range(0, weight_matrix.shape[0]), 200)

    cached_file = "Results/cached_results.p"
    if(os.path.exists(cached_file)):
        cached_results = pickle.load(open(cached_file, "rb"))
    else:
        cached_results = {}
        for param in cached_params:
            query_size, cache_size, alpha, norm_method = param
            cached_vectors = io_utils.load_cached_vectors(db_file, query_nodes[:query_size], alpha, cache_size)
            v, i, e = ppr.cached_ppr(weight_matrix, cached_vectors, alpha, norm_method=norm_method)
            cached_results[param] = i
        pickle.dump(cached_results, open(cached_file, "wb"))

    standard_file = "Results/standard_results.p"
    if(os.path.exists(standard_file)):
        standard_results = pickle.load(open(standard_file, "rb"))
    else:
        standard_results = {}
        for param in standard_params:
            query_size, alpha = param
            v, i, e = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
            standard_results[param] = i

        pickle.dump(standard_results, open(standard_file, "wb"))

    return cached_results, standard_results


def generic_line_plot(xlabel="Cache Size", ylabel= "Number of Iterations", title = ""):
    return None


def make_all_plots():
    plot_alphas()
    plot_cache_sizes()
    plot_query_sizes()


def main():
    cached_results, standard_results = janky_results("Data/Email-Enron.mat")
    print(cached_results, standard_results)

main()
