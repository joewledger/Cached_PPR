from scipy.sparse import *
from scipy.sparse.linalg import *
import numpy as np
import vector_utils as vu


class PPR_Result:

    def __init__(self, final_vector, num_iterations, error_terms):
        self.final_vector = final_vector
        self.num_iterations = num_iterations
        self.error_terms = error_terms


def ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000

    iterations = 0
    vectors = [start_vector]
    error_terms = [1.0]

    while(error_terms[-1] > eps and iterations < max_iter):

        vectors.append(calculate_next_vector(weight_matrix, vectors[-1], restart_vector, alpha))
        error_terms.append(get_l1_norm(vectors[-1], vectors[-2]))
        vectors = vectors[1:]
        iterations += 1

    return PPR_Result(vectors[-1], iterations, error_terms[1:])


def chebyshev_ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000
    iterations = 0
    error_terms = [1.0]

    dimension = weight_matrix.shape[0]

    mu_values = [0.0, - (1.0 / (1.0 - alpha))]
    vectors = [zeroes_vector(dimension), start_vector]

    while(error_terms[-1] > eps and iterations < max_iter):

        mu_values, vectors = chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values)
        error_terms.append(get_l1_norm(vectors[-1], vectors[-2]))
        mu_values = mu_values[1:]
        vectors = vectors[1:]
        iterations += 1

    return PPR_Result(vectors[-1], iterations, error_terms[1:])


def standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr):
    restart_vector = get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


def cached_ppr(weight_matrix, query_nodes, vector_cache, cache_size, alpha, ppr_method=ppr, norm_method=vu.unnormalized):
    dimension = weight_matrix.shape[0]

    vector_list = vector_cache.get_vector_list(query_nodes, alpha, cache_size=cache_size)
    start_vector = norm_method(vector_list)
    restart_vector = get_restart_vector(dimension, query_nodes)

    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


def get_restart_vector(dimension, query_nodes):
    entries = {(x, 0): 1 / len(query_nodes) for x in query_nodes}

    matrix = dok_matrix((dimension, 1))
    matrix.update(entries)
    return matrix.tocsr()


def zeroes_vector(row_dim):
    return csr_matrix((row_dim, 1))


def get_l1_norm(current_vector, previous_vector):
    return abs(current_vector - previous_vector).sum(0).item(0)


def calculate_next_vector(weight_matrix, curr_vector, restart_vector, alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector


def chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values):
    mu_values.append(2.0 / (1.0 - alpha) * mu_values[-1] - mu_values[-2])
    first_product = 2.0 * (mu_values[-2] / mu_values[-1]) * weight_matrix.dot(vectors[-1])
    second_product = (mu_values[-3] / mu_values[-1]) * vectors[-2]
    third_product = (2.0 * mu_values[-2]) / ((1.0 - alpha) * mu_values[-1]) * alpha * restart_vector
    vectors.append(first_product - second_product + third_product)
    return mu_values, vectors


def trim_vector(vector, k):
    data = vector.toarray().flatten()
    ind = np.argpartition(data, -k)[-k:]
    matrix = dok_matrix(vector.shape)
    entries = {(i, 0): data[i] for i in ind}
    matrix.update(entries)
    return matrix.tocsr()


def get_proximity_vector(weight_matrix, query_node, alpha, ppr_method=chebyshev_ppr, eps=1E-10):
    dimension = weight_matrix.shape[0]
    restart_vector = get_restart_vector(dimension, [query_node])
    start_vector = restart_vector.copy()
    proximity_vector = ppr_method(weight_matrix, start_vector, restart_vector, alpha, eps=eps).final_vector
    return proximity_vector
