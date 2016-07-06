import ppr
import argparse
import io_utils
import itertools
from joblib import Parallel, delayed


def build_cache(network_filepath, db_filepath, alphas, num_threads=5, top_k=200, cache_subdir="Cache/", subset_size=None):
    weight_matrix = io_utils.load_csr_matrix(network_filepath)
    dimension = weight_matrix.shape[0]

    db_wrapper = io_utils.DBWrapper(db_filepath)
    db_wrapper.open_connection()
    db_wrapper.initialize_all_tables()
    proximity_filepaths = db_wrapper.get_unique_proximity_filepaths(count=dimension * len(alphas), path_prefix=cache_subdir)
    alpha_mapping = db_wrapper.get_alpha_mapping_and_update_table(alphas)
    db_wrapper.close_connection()
    alpha_ids = list(alpha_mapping.keys())

    parameters = zip(proximity_filepaths, *[itertools.product(range(dimension), alpha_ids)])
    results = Parallel(n_jobs=num_threads)(delayed(compute_and_save_proximity_vector)
                      (weight_matrix, alpha_mapping, p, top_k) for p in parameters)

    db_wrapper.open_connection()
    for query_node, alpha_id, proximity_filepath in results:
        db_wrapper.add_proximity_vector_filepath(network_filepath, query_node, alpha_id, proximity_filepath, dimension)
    db_wrapper.close_connection()

def compute_and_save_proximity_vector(weight_matrix, alpha_mapping, params, k):
    proximity_filepath, param_tuple = params
    query_node, alpha_id = param_tuple
    proximity_vector = ppr.get_proximity_vector(weight_matrix, query_node, alpha_mapping[alpha_id], top_k=k)
    io_utils.pickle_proximity_vector(proximity_filepath, proximity_vector)
    return (query_node, alpha_id, proximity_filepath)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds cache of stored proximity vectors")
    parser.add_argument('--network_filepath', type=str)
    parser.add_argument('--db_filepath', type=str)
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--cache_subdir', type=str)
    parser.set_defaults(db_filepath="Cache/proximity_vectors.sqlite3", network_filepath="Data/Email-Enron.mat",
                        alphas=[.01, .1, .25], num_threads=5, top_k=200, cache_subdir="Cache/")
    args = parser.parse_args()

    kwargs = dict(num_threads=args.num_threads, top_k=args.top_k, cache_subdir=args.cache_subdir)

    build_cache(args.network_filepath, args.db_filepath, args.alphas, **kwargs)
