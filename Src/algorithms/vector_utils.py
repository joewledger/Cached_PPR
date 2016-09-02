import random
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *


#Creates a list of num_sets query sets, each of which is of size set_size, and comes from population
#Example call: get_query_sets(4, 100, range(1000)) -> 4 query sets of size 100 from the population 0, 1000
def get_query_sets(num_sets, set_size, population):
    return [random.sample(population, set_size) for _ in range(num_sets)]


#Returns the sum of a list of vectors
def unnormalized(vector_list):
    return sum(vector_list)


#Returns the sum of a list of vectors normalized by the total sum of the resultant vector
def total_sum(vector_list):
    vector = sum(vector_list)
    return vector / vector.sum()


#Returns the sum of a list of normalized vectors, which is then normalized by itself.
def twice_normalized(vector_list):
    normalized_vectors = [v / v.sum() for v in vector_list]
    vector = sum(normalized_vectors)
    return vector / vector.sum()


#Gets the standard restart vector in the form of a CSR sparse matrix for a set of query nodes
def get_restart_vector(dimension, query_nodes):
    entries = {(x, 0): 1 / len(query_nodes) for x in query_nodes}
    matrix = dok_matrix((dimension, 1))
    matrix.update(entries)
    return matrix.tocsr()


#Returns a zeroes vector in CSR format of dimension [row_dim, 1]
def zeroes_vector(row_dim):
    return csr_matrix((row_dim, 1))


#Gets the L1-distance between two vectors
def get_l1_distance(current_vector, previous_vector):
    return get_l1_norm(current_vector - previous_vector)


#Gets the L1-norm of a vector
def get_l1_norm(vector):
    return abs(vector).sum(0).item(0)


#Returns a CSR matrix [m:1] of a CSR matrix [m:1] trimmed to its top k elements
def trim_vector(vector, k):
    data = vector.toarray().flatten()
    ind = np.argpartition(data, -k)[-k:]
    matrix = dok_matrix(vector.shape)
    entries = {(i, 0): data[i] for i in ind}
    matrix.update(entries)
    return matrix.tocsr()


#Given a vector and a list of indices to keep, this method will return a CSR vector containing only these indices
def build_vector_subset(vector, entry_indices):
    vector_subset = dok_matrix(vector.shape)
    vector_subset.update({(x, 0): vector[x, 0] for x in entry_indices})
    return vector_subset.tocsr()


#Returns the k_th value of a CSR sparse matrix
def kth_value(vector, k):
    data = vector.toarray().flatten()
    ind = np.argpartition(data, -k)[-k]
    return data[ind]


#Returns a set containing all the row indices of a vector that contain non-zero entries
def get_nonzero_indices_set(vector):
    return set(vector.nonzero()[0])


def get_maximum_value(vector):
    return vector.max()
