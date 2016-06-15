
from scipy.sparse import *
import scipy.io as spio
from scipy.sparse.linalg import *
import numpy as np
import random
import pickle
import os
import multiprocessing
#dok_matrix -> (dictionary of keys matrix) used for constructing sparse matrices incrementally
#csr_matrix -> (compressed sparse row matrix) good for fast matrix arithmetic and matrix-vector products


def read_csr_matrix(filename):
    return spio.loadmat(filename)["normalizedNetwork"]

def generate_random_start_vector(dimension,k):
    
    query_nodes = random.sample(range(0,dimension),k)
    return query_nodes,get_start_vector(dimension,query_nodes)

def get_start_vector(dimension,query_nodes):
    entries = {(x,0) : 1 / len(query_nodes) for x in query_nodes}

    matrix = dok_matrix((dimension,1))
    matrix.update(entries)
    return matrix.tocsr()


def calculate_next_vector(weight_matrix,curr_vector,restart_vector,alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector

def cached_ppr(weight_matrix,cache_name, query_nodes,alpha):
    return None


def ppr(weight_matrix, start_vector, restart_vector, alpha):

    eps = .0001
    max_iter = 1000

    iterations = 1
    prev_vector = restart_vector.copy()
    curr_vector = calculate_next_vector(weight_matrix,prev_vector,restart_vector,alpha)

    l1_norm = lambda c,p : norm(c-p)
    error_term = l1_norm(curr_vector,prev_vector)

    while(error_term > eps and iterations < max_iter):
        prev_vector = curr_vector.copy()
        curr_vector = calculate_next_vector(weight_matrix,curr_vector,restart_vector,alpha)
        error_term = l1_norm(curr_vector,prev_vector)
        iterations += 1

    return curr_vector,iterations

        
def print_top_k_indices_and_scores(final_vector,k):
    data = final_vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    for index in indices:
        print(index,final_vector[index,0])


if __name__ == "__main__":

    
    weight_matrix = read_csr_matrix("Data\%s.mat" % cache_name)
    alpha = .5

    #build_cache(cache_name,weight_matrix,alpha,num_threads=2)
    
    vector = pickle.load(open(r"Cache\Email-Enron\alpha0.5\cache0.p","rb"))
    print_top_k_indices_and_scores(vector,5)

    start_vector = get_start_vector(weight_matrix.shape[0],[0])
    vector2,iterations = ppr(weight_matrix, start_vector, start_vector.copy(), alpha)
    print_top_k_indices_and_scores(vector2,5)