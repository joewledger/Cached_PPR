import argparse
import vector_cache as vc
import vector_utils as vu
import io_utils as io
import pickle


def save_cache(network_filepath, cache_filepath, query_set_filepath, alphas, num_queries):
    weight_matrix = io.load_csr_matrix(network_filepath)

    query_set = vu.get_query_sets(1, num_queries, range(weight_matrix.shape[0]))[0]

    cache = vc.vector_cache()
    cache.build_cache(weight_matrix, query_set, alphas)

    cache.save_to_file(cache_filepath)
    pickle.dump(query_set, open(query_set_filepath, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves vector cache for use in future experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--cache_filepath', type=str)
    parser.add_argument('--query_set_filepath', type=str)
    parser.add_argument('--num_queries', type=int)
    parser.add_argument('--alphas', type=float, nargs='+')

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", cache_filepath="Cache/Email-Enron.pickle",
                        query_set_filepath="Cache/Email-Enron-queries.pickle", alphas=[.01, .1, .25],
                        num_queries=2000)

    args = parser.parse_args()

    save_cache(args.network_filepath, args.cache_filepath, args.query_set_filepath, args.alphas, args.num_queries)
