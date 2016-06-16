
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

def generate_random_start_vector(dimension,num_queries):
    
    query_nodes = random.sample(range(0,dimension),num_queries)
    return query_nodes,get_restart_vector(dimension,query_nodes)

def get_restart_vector(dimension,query_nodes):
    entries = {(x,0) : 1 / len(query_nodes) for x in query_nodes}

    matrix = dok_matrix((dimension,1))
    matrix.update(entries)
    return matrix.tocsr()

def get_cached_start_vector(dimension,cached_vectors,cache_size,norm_method=None):
    entries = {}

    for vector in cached_vectors:
        curr_vector_cache = get_top_k_index_value_tuples(vector,cache_size)
        for entry in curr_vector_cache:
            key_tuple = entry[0],0
            value = entry[1]
            entries[key_tuple] = (entries[key_tuple] + value if key_tuple in entries else value)

    if(norm_method):
        entries = normalize_entries_dict(entries,len(cached_vectors),method=norm_method)
    vector = dok_matrix((dimension, 1))
    vector.update(entries)
    return vector.tocsc()

def normalize_entries_dict(entries,num_queries,method="total_sum"):
    if(method == "total_sum"):
        total_sum = sum(v for v in entries.values())
        return {key : entries[key] / total_sum for key in entries.keys()}
    elif(method == "num_queries"):
        return {key : entries[key] / num_queries for key in entries.keys()}


def cached_ppr(weight_matrix,cache_path, query_nodes,alpha,cache_size,norm_method="total_sum"):

    dimension = weight_matrix.shape[0]

    cached_vectors = [pickle.load(open("%s%d.p" % (cache_path,i),"rb")) for i in query_nodes]
    start_vector = get_cached_start_vector(dimension,cached_vectors,cache_size,
                                           norm_method=norm_method)
    restart_vector = get_restart_vector(dimension,query_nodes)
    
    return ppr(weight_matrix,start_vector,restart_vector,alpha)

def generic_ppr(weight_matrix,query_nodes,alpha):
    restart_vector = get_restart_vector(weight_matrix.shape[0],query_nodes)
    start_vector = restart_vector.copy()
    return ppr(weight_matrix,start_vector,restart_vector,alpha)

def ppr(weight_matrix, start_vector, restart_vector, alpha):

    eps = .0001
    max_iter = 1000

    iterations = 1
    prev_vector = start_vector.copy()
    curr_vector = calculate_next_vector(weight_matrix,prev_vector,restart_vector,alpha)

    l1_norm = lambda c,p : norm(c-p)
    error_term = l1_norm(curr_vector,prev_vector)

    while(error_term > eps and iterations < max_iter):
        prev_vector = curr_vector.copy()
        curr_vector = calculate_next_vector(weight_matrix,curr_vector,restart_vector,alpha)
        error_term = l1_norm(curr_vector,prev_vector)
        iterations += 1

    return curr_vector,iterations

def calculate_next_vector(weight_matrix,curr_vector,restart_vector,alpha):
    return (1 - alpha) * weight_matrix.dot(curr_vector) + alpha * restart_vector

def get_top_k_index_value_tuples(final_vector,k):
    data = final_vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    return [(indices[i],data[indices[i]]) for i in range(0,k)]
        
def print_top_k_indices_and_scores(final_vector,k):
    data = final_vector.toarray().flatten()
    indices = np.argsort(data)[-k:]
    for index in indices:
        print(index,final_vector[index,0])