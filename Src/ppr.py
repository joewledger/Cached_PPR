from scipy.sparse import *
from scipy.sparse.linalg import *
import numpy as np
import vector_utils as vu
import math


class PPR_Result:

    def __init__(self, final_vector, num_iterations, error_terms):
        self.final_vector = final_vector
        self.num_iterations = num_iterations
        self.error_terms = error_terms


#Standard implementation of the PPR algorithm.
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


def chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10):
    max_iter = 10000
    iterations = 0

    Kappa = (2-alpha)/alpha;
    Xi = (math.sqrt(Kappa)-1)/(math.sqrt(Kappa) +1);
    ErrorBound = Xi;

    muPPrevious = 1.0;
    muPrevious = 1/(1-alpha);
    mu = 0.0;

    dimension = weight_matrix.shape[0]

    mPPreviousScore = zeroes_vector(dimension)
    mPreviousScore = start_vector
    mScore = zeroes_vector(dimension)

    error_terms = [1.0]

    r = np.arange(dimension)
    while(r.size > k):
        mu = 2 / (1 - alpha) * muPrevious - muPPrevious
        mScore = 2 * (muPrevious / mu) * weight_matrix.dot(mPreviousScore) - (muPPrevious / mu) * mPPreviousScore + (2 * muPrevious) / ((1 - alpha) * mu) * alpha * restart_vector
        theta = kth_value(mScore, k)
        f = mScore.toarray().flatten()
        r = np.argwhere(f + 4 * ErrorBound > theta)
        iterations += 1
        ErrorBound = ErrorBound * Xi
        muPPrevious = muPrevious
        muPrevious = mu
        mPPreviousScore = mPreviousScore
        mPreviousScore = mScore
        if(iterations == max_iter or ErrorBound < eps):
            break
    final_vector = dok_matrix((dimension, 1))
    final_vector.update({(x, 0) : mScore[x,0] for x in r.flatten()})
    return PPR_Result(final_vector, iterations, None)


def indexed_chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10):
    max_iter = 10000
    iterations = 0
    error_terms = []

    Kappa = (2-alpha)/alpha;
    Xi = (math.sqrt(Kappa)-1)/(math.sqrt(Kappa) +1);
    ErrorBound = Xi ** 2;

    dimension = weight_matrix.shape[0]

    mu_values = [0.0, - (1.0 / (1.0 - alpha))]
    vectors = [zeroes_vector(dimension), start_vector]

    error_adjusted = False
    log_threshold = .1
    ratio = lambda e : (e[-3] / e[-2]) - (e[-2]/ e[-1])
    unbiased_vectors = [zeroes_vector(dimension), restart_vector.copy()]
    unbiased_mu_values, unbiased_vectors = chebyshev_next_iteration(weight_matrix, restart_vector, alpha, unbiased_vectors, mu_values)
    unbiased_error = get_l1_norm(unbiased_vectors[-1], unbiased_vectors[-2])
    print(unbiased_error)

    r = np.arange(dimension)
    while(r.size > k and iterations < max_iter and ErrorBound > eps):

        mu_values, vectors = chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values)
        error_terms.append(get_l1_norm(vectors[-1], vectors[-2]))
        mu_values = mu_values[1:]
        vectors = vectors[1:]
        if(len(error_terms) > 2):
            print(ratio(error_terms))


        theta = kth_value(vectors[-2], k)
        f = vectors[-1].toarray().flatten()
        r = np.argwhere(f + 4 * ErrorBound > theta)
        ErrorBound = ErrorBound * Xi
        iterations += 1

    final_vector = dok_matrix((dimension, 1))
    final_vector.update({(x, 0) : vectors[-1][x,0] for x in r.flatten()})

    return PPR_Result(final_vector, iterations, error_terms[1:])


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





def calculate_next_vector(weight_matrix, curr_vector, restart_vector, alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector


def chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values):
    mu_values.append(2.0 / (1.0 - alpha) * mu_values[-1] - mu_values[-2])
    first_product = 2.0 * (mu_values[-2] / mu_values[-1]) * weight_matrix.dot(vectors[-1])
    second_product = (mu_values[-3] / mu_values[-1]) * vectors[-2]
    third_product = (2.0 * mu_values[-2]) / ((1.0 - alpha) * mu_values[-1]) * alpha * restart_vector
    vectors.append(first_product - second_product + third_product)
    return mu_values, vectors


def get_proximity_vector(weight_matrix, query_node, alpha, ppr_method=chebyshev_ppr, eps=1E-10):
    dimension = weight_matrix.shape[0]
    restart_vector = get_restart_vector(dimension, [query_node])
    start_vector = restart_vector.copy()
    proximity_vector = ppr_method(weight_matrix, start_vector, restart_vector, alpha, eps=eps).final_vector
    return proximity_vector
