from scipy.sparse import *
from scipy.sparse.linalg import *
import numpy as np
import math

import start_vectors as sv


class PPR_Result:

    def __init__(self, final_vector, num_iterations, error_terms):
        self.final_vector = final_vector
        self.num_iterations = num_iterations
        self.error_terms = error_terms


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

    return PPR_Result(curr_vector, iterations, error_terms[1:])


def chebyshev_ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000
    dimension = weight_matrix.shape[0]

    c = .5
    muPPrevious = 1.0
    muPrevious = 1.0 / (1.0 - c)
    mu = 0.0

    iterations = 0
    error_terms = [1.0]
    #myK = (2.0 - c) / c
    #myXi = (math.sqrt(myK) - 1.0) / (math.sqrt(myK) + 1.0)

    mScore = zeroes_vector(dimension)
    mPPreviousScore = zeroes_vector(dimension)
    mPreviousScore = start_vector / 2

    while(error_terms[-1] > eps and iterations < max_iter):
        mPPreviousScore = mPreviousScore
        mPreviousScore = mScore

        muPPrevious = muPrevious
        muPrevious = mu

        mu = 2.0 / (1.0 - c) * muPrevious - muPPrevious
        first_product = 2.0 * (muPrevious / mu) * weight_matrix.dot(mPreviousScore)
        second_product = (muPPrevious / mu) * mPPreviousScore
        third_product = (2.0 * muPrevious) / ((1.0 - c) * mu) * c * restart_vector
        mScore = first_product - second_product + third_product

        error_terms.append(get_l1_norm(mScore, mPreviousScore))
        iterations += 1

    return PPR_Result(mScore / mScore.sum(), iterations, error_terms)


def standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr):
    restart_vector = get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


def cached_ppr(weight_matrix, query_nodes, vector_cache, cache_size, alpha, norm_method=sv.unnormalized):
    dimension = weight_matrix.shape[0]

    vector_list = vector_cache.get_vector_list(query_nodes, alpha, cache_size=cache_size)
    start_vector = norm_method(vector_list)
    restart_vector = get_restart_vector(dimension, query_nodes)

    return ppr(weight_matrix, start_vector, restart_vector, alpha)


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


def trim_vector(vector, k):
    data = vector.toarray().flatten()
    ind = np.argpartition(data, -k)[-k:]
    matrix = dok_matrix(vector.shape)
    entries = {(i, 0): data[i] for i in ind}
    matrix.update(entries)
    return matrix.tocsr()


def get_proximity_vector(weight_matrix, query_node, alpha, eps=1E-10):
    dimension = weight_matrix.shape[0]
    restart_vector = get_restart_vector(dimension, [query_node])
    start_vector = restart_vector.copy()
    proximity_vector = ppr(weight_matrix, start_vector, restart_vector, alpha, eps=eps).final_vector
    return proximity_vector
