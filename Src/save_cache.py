import argparse
import vector_cache as vc
import vector_utils as vu
import io_utils as io
import pickle


def save_cache(network_filepath, alphas, num_queries):

    cache_filepath = io.get_cache_filepath(network_filepath)
    query_set_filepath = io.get_cached_queries_filepath(network_filepath)

    weight_matrix = io.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]

    query_set = vu.get_query_sets(1, min(dimension, num_queries), range(dimension))[0]

    cache = vc.vector_cache()
    cache.build_cache(weight_matrix, query_set, alphas, eps=1E-5)

    cache.save_to_file(cache_filepath)
    pickle.dump(query_set, open(query_set_filepath, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves vector cache for use in future experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--num_queries', type=int)
    parser.add_argument('--alphas', type=float, nargs='+')

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", alphas=[.01, .1, .25],
                        num_queries=2000)

    args = parser.parse_args()

    save_cache(args.network_filepath, args.alphas, args.num_queries)
