
from scipy.sparse import *
from scipy.sparse.linalg import *
import numpy as np


def standard_ppr(weight_matrix, query_nodes, alpha):
    restart_vector = get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return ppr(weight_matrix, start_vector, restart_vector, alpha)


# cached_vectors = {first_node -> {second_node -> score}}
def cached_ppr(weight_matrix, cached_vectors, alpha, norm_method="total_sum"):
    dimension = weight_matrix.shape[0]

    start_vector = build_cached_start_vector(
        dimension, cached_vectors, norm_method=norm_method)
    restart_vector = get_restart_vector(dimension, list(cached_vectors.keys()))

    return ppr(weight_matrix, start_vector, restart_vector, alpha)


# cached_vectors = {first_node -> {second_node -> score}}
# returns a matrix
def build_cached_start_vector(dimension, cached_vectors, norm_method=None):
    normalization_dict = {"total_sum": total_sum_cached_start_vector,
                          "twice_normalized": twice_normalized_cached_start_vector}
    normalization_function = normalization_dict.get(
        norm_method, unnormalized_cached_start_vector)
    entries = normalization_function(cached_vectors)
    vector_entries = {(key, 0): entries[key] for key in entries.keys()}

    start_vector = dok_matrix((dimension, 1))
    start_vector.update(vector_entries)
    return start_vector.tocsr()


def unnormalized_cached_start_vector(cached_vectors):
    entries = {}
    for fn in cached_vectors.keys():
        for sn in cached_vectors[fn].keys():
            entries[sn] = (cached_vectors[fn][sn] + (entries[sn] if sn in entries else 0.0))
    return entries


def total_sum_cached_start_vector(cached_vectors):
    entries = {}
    for fn in cached_vectors.keys():
        for sn in cached_vectors[fn].keys():
            value = float(cached_vectors[fn][sn])
            entries[sn] = (value + (float(entries[sn])
                                    if sn in entries else 0.0))

    total_sum = float(sum(entries.values()))
    for fn in entries.keys():
        entries[fn] = float(entries[fn]) / total_sum
    return entries


def twice_normalized_cached_start_vector(cached_vectors):
    entries = {}
    for fn in cached_vectors.keys():
        node_sum = sum(cached_vectors[fn].values())
        for sn in cached_vectors[fn].keys():
            value = float(cached_vectors[fn][sn]) / node_sum
            entries[sn] = (value + (float(entries[sn]) if sn in entries else 0.0))

    total_sum = float(sum(entries.values()))
    for fn in entries.keys():
        entries[fn] = float(entries[fn]) / total_sum
    return entries


def get_restart_vector(dimension, query_nodes):
    entries = {(x, 0): 1 / len(query_nodes) for x in query_nodes}

    matrix = dok_matrix((dimension, 1))
    matrix.update(entries)
    return matrix.tocsr()


def ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000

    iterations = 0
    curr_vector = start_vector.copy()

    error_terms = [1.0]

    while(error_terms[-1] > eps and iterations < max_iter):

        prev_vector = curr_vector.copy()
        curr_vector = calculate_next_vector(weight_matrix, curr_vector, restart_vector, alpha)
        error_terms.append(get_l1_norm(curr_vector, prev_vector))
        iterations += 1

    return curr_vector, iterations, error_terms[1:]


def get_l1_norm(current_vector, previous_vector):
    return abs(current_vector - previous_vector).sum(0).item(0)


def calculate_next_vector(weight_matrix, curr_vector, restart_vector, alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector


def get_top_k_index_value_tuples(final_vector, k):
    data = final_vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    return [(int(indices[i]), data[indices[i]]) for i in range(0, k)]


def print_top_k_indices_and_scores(final_vector, k):
    data = final_vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    for index in indices:
        print(index, final_vector[index, 0])


def limit_vector_top_k(vector, k):
    data = vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    matrix = dok_matrix(vector.shape)
    entries = {(index, 0): data[index] for index in indices}
    matrix.update(entries)
    return matrix.tocsr()


def get_proximity_vector(weight_matrix, query_node, alpha, eps=1E-10, top_k=None):
    dimension = weight_matrix.shape[0]
    restart_vector = get_restart_vector(dimension, [query_node])
    start_vector = restart_vector.copy()
    curr_vector, iterations, error_terms = ppr(weight_matrix, start_vector, restart_vector, alpha, eps=eps)
    if(top_k):
        curr_vector = limit_vector_top_k(curr_vector, top_k)
    return curr_vector
