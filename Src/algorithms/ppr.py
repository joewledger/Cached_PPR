import algorithms.vector_utils as vu
from scipy.sparse.linalg import *
import numpy as np
import math


"""
This module provides several algorithms for computing PPR queries and top-k queries.
These include the standard PPR algorithm, Chebyshev PPR algorithm, and Chebyshev top-K query algorithm.

For the global PPR queries, you need to provide a weight matrix, start vector, restart vector, and alpha value.

For the top-K queries, you will need to provide a weight_matrix, start vector, restart vector, alpha value, and k value.

A default error threshold epsilon of 1E-10 is assumed, but can be changed using the keyword argument.

If you have a set of query nodes that you want to compute a query for, or you want to utilize indexing to speed queries,
consider using the ppr_interface module.
"""


#Provides a simple wrapper for aggregating the results and runtime characteristics of a PPR query
class PPR_Result:

    def __init__(self, final_vector, num_iterations, error_terms):
        self.final_vector = final_vector
        self.num_iterations = num_iterations
        self.error_terms = error_terms


#Standard implementation of the PPR algorithm.
def standard_ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000

    iterations = 0
    vectors = [start_vector]
    error_terms = [1.0]

    #Power iteration continues until the L1-distance between iterations is below
    #the threshold epsilon, or the maximum number of iterations is reached
    while(error_terms[-1] > eps and iterations < max_iter):

        vectors.append(calculate_next_vector(weight_matrix, vectors[-1], restart_vector, alpha))
        error_terms.append(vu.get_l1_distance(vectors[-1], vectors[-2]))
        vectors = vectors[1:]
        iterations += 1

    return PPR_Result(vectors[-1], iterations, error_terms[1:])


#Calculates the next value of the node weight vector for the standard PPR algorithm
def calculate_next_vector(weight_matrix, curr_vector, restart_vector, alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector


#Chebyshev implementation of the PPR algorithm
def chebyshev_ppr(weight_matrix, start_vector, restart_vector, alpha, eps=1E-10):
    max_iter = 10000
    iterations = 0
    error_terms = [1.0]

    dimension = weight_matrix.shape[0]

    mu_values = [0.0, - (1.0 / (1.0 - alpha))]
    vectors = [vu.zeroes_vector(dimension), start_vector]

    #Power iteration continues until the L1-distance between iterations is below
    #the threshold epsilon, or the maximum number of iterations is reached
    while(error_terms[-1] > eps and iterations < max_iter):

        mu_values, vectors = chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values)
        error_terms.append(vu.get_l1_distance(vectors[-1], vectors[-2]))
        mu_values = mu_values[1:]
        vectors = vectors[1:]
        iterations += 1

    return PPR_Result(vectors[-1], iterations, error_terms[1:])


#Calculates the next value of node weight vector for the chebyshev PPR algorithm
def chebyshev_next_iteration(weight_matrix, restart_vector, alpha, vectors, mu_values):
    mu_values.append(2.0 / (1.0 - alpha) * mu_values[-1] - mu_values[-2])
    first_product = 2.0 * (mu_values[-2] / mu_values[-1]) * weight_matrix.dot(vectors[-1])
    second_product = (mu_values[-3] / mu_values[-1]) * vectors[-2]
    third_product = (2.0 * mu_values[-2]) / ((1.0 - alpha) * mu_values[-1]) * alpha * restart_vector
    vectors.append(first_product - second_product + third_product)
    return mu_values, vectors


#Implementation of the chebyshev top K algorithm
def chebyshev_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10):
    max_iter = 10000
    iterations = 0

    Kappa = (2 - alpha) / alpha
    Xi = (math.sqrt(Kappa) - 1) / (math.sqrt(Kappa) + 1)
    ErrorBound = Xi * vu.get_maximum_value(start_vector)
    print(ErrorBound)

    muPPrevious = 1.0
    muPrevious = 1 / (1 - alpha)
    mu = 0.0

    dimension = weight_matrix.shape[0]

    mPPreviousScore = vu.zeroes_vector(dimension)
    mPreviousScore = start_vector
    mScore = vu.zeroes_vector(dimension)

    r = np.arange(dimension)

    while(r.size > k and iterations < max_iter and ErrorBound > eps):
        mu = 2 / (1 - alpha) * muPrevious - muPPrevious
        mScore = 2 * (muPrevious / mu) * weight_matrix.dot(mPreviousScore) - (muPPrevious / mu) * mPPreviousScore + (2 * muPrevious) / ((1 - alpha) * mu) * alpha * restart_vector
        theta = vu.kth_value(mScore, k)
        f = mScore.toarray().flatten()
        r = np.argwhere(f + 4 * ErrorBound > theta)
        iterations += 1
        ErrorBound = ErrorBound * Xi
        muPPrevious = muPrevious
        muPrevious = mu
        mPPreviousScore = mPreviousScore
        mPreviousScore = mScore

    final_vector = vu.build_vector_subset(mScore, r.flatten())
    return PPR_Result(final_vector, iterations, None)


def squeeze_top_k(weight_matrix, start_vector, restart_vector, alpha, k, eps=1E-10):
    max_iter = 10000
    iterations = 0

    ErrorBound = (1 - alpha) * vu.get_maximum_value(start_vector)

    dimension = weight_matrix.shape[0]

    mPreviousScore = start_vector
    mScore = vu.zeroes_vector(dimension)

    r = np.arange(dimension)

    #Power iteration continues until set of final top nodes can be trimmed below k
    while(r.size > k and iterations < max_iter and ErrorBound > eps):
        mScore = (1 - alpha) * weight_matrix.dot(mPreviousScore) + alpha * restart_vector
        theta = vu.kth_value(mScore, k)
        f = mScore.toarray().flatten()
        r = np.argwhere(f + ErrorBound > theta)
        iterations += 1
        ErrorBound = ErrorBound * (1 - alpha)
        mPreviousScore = mScore

    final_vector = vu.build_vector_subset(mScore, r.flatten())
    return PPR_Result(final_vector, iterations, None)
