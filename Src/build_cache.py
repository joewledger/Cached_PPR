import ppr
import argparse
import io_utils


def build_cache(network_filepath, db_filepath, alphas, num_threads=5, top_k=200, max_dim=None, cache_subdir="Cache/"):
    weight_matrix = io_utils.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]
    if(max_dim):
        dimension = min(dimension, max_dim)

    db_wrapper = io_utils.DBWrapper(db_filepath)
    db_wrapper.open_connection()
    db_wrapper.initialize_all_tables()

    for alpha in alphas:
        alpha_id = db_wrapper.get_closest_alpha_tuple_and_update_table(alpha)[0]

        proximity_filepaths = db_wrapper.get_unique_proximity_filepaths(count=dimension, path_prefix=cache_subdir)
        proximity_vectors = {i: ppr.get_proximity_vector(weight_matrix, i, alpha, top_k=top_k) for i in range(0, dimension)}
        query_node_to_proximity_filepath = {i: proximity_filepaths[i] for i in list(proximity_vectors.keys())}
        for index, vector in proximity_vectors.items():
            io_utils.pickle_proximity_vector(proximity_filepaths[index], vector)
        db_wrapper.update_proximity_vector_filepaths(network_filepath, query_node_to_proximity_filepath, alpha_id, dimension)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds cache of stored proximity vectors")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--db_filepath', type=str)
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--max_dim', type=int)
    parser.add_argument('--cache_subdir', type=str)
    parser.set_defaults(db_filepath="Cache/proximity_vectors.sqlite3", network_filepath="Data/Email-Enron.mat",
                        alphas=[.01, .1, .25], num_threads=5, top_k=200, cache_subdir="Cache/")
    args = parser.parse_args()

    kwargs = dict(num_threads=args.num_threads, top_k=args.top_k, max_dim=args.max_dim, cache_subdir=args.cache_subdir)

    build_cache(args.network_filepath, args.db_filepath, args.alphas, **kwargs)
