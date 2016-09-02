import argparse
import vector_index as vc
import vector_utils as vu
import io_utils as io
import pickle


def save_index(network_filepath, alphas, num_queries, num_processes):

    index_filepath = io.get_index_filepath(network_filepath)
    query_set_filepath = io.get_indexd_queries_filepath(network_filepath)

    weight_matrix = io.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]

    query_set = vu.get_query_sets(1, min(dimension, num_queries), range(dimension))[0]

    index = vc.vector_index()
    index.build_index(weight_matrix, query_set, alphas, eps=1E-10, num_processes=num_processes)

    index.save_to_file(index_filepath)
    pickle.dump(query_set, open(query_set_filepath, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves vector index for use in future experiments")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--num_queries', type=int)
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--num_processes', type=int)

    parser.set_defaults(network_filepath="Data/Email-Enron.mat", alphas=[.01, .1, .25],
                        num_queries=2000, num_processes=10)

    args = parser.parse_args()

    save_index(args.network_filepath, args.alphas, args.num_queries, args.num_processes)
