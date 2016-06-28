
import ppr
import scipy as sp
import random
import io_utils

def test_standard_ppr():
    weight_matrix = io_utils.read_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    restart_vector = ppr.get_restart_vector(weight_matrix.shape[0],random.sample(range(0,dimension), 5))
    start_vector = restart_vector.copy()
    alpha = .01

    v1, i1, e1 = ppr.ppr(weight_matrix, start_vector, restart_vector, alpha)
    print(v1, i1, e1)

def test_cached_ppr():
    weight_matrix = io_utils.read_csr_matrix("Data/Email-Enron.mat")
    alpha = .01
    query_nodes = [2, 4]
    cache_size = 100
    cached_vectors = get_generic_cached_vectors(alpha=alpha, query_nodes=query_nodes, cache_size=100)

    v1, i1, e1 = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
    
    v2, i2, e2 = ppr.cached_ppr(weight_matrix, cached_vectors, alpha, norm_method="total_sum")

    print(i1, i2)

def test_scalar_multiply():
    weight_matrix = io_utils.read_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    restart_vector = ppr.get_restart_vector(weight_matrix.shape[0],random.sample(range(0,dimension), 5))
    start_vector = restart_vector.copy()
    alpha = .01

    def scalar_experiment(weight_matrix, alpha, start_vector,restart_vector, c, d):
        return ppr.ppr(weight_matrix, start_vector.multiply(c), restart_vector.multiply(d), alpha)

    v1, i1, e1 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,1.0,1.0)
    v2, i2, e2 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,5.0,1.0)
    v3, i3, e3 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,1.0,5.0)
    v4, i4, e4 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,5.0,5.0)

    print(i1,i2,i3,i4)

    print(e1[-100:])

def test_l1_norm():

    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    start_vector = ppr.get_restart_vector(weight_matrix.shape[0],random.sample(range(0,dimension), 5))
    restart_vector = ppr.get_restart_vector(weight_matrix.shape[0],random.sample(range(0,dimension), 5))

    print(ppr.l1_norm(start_vector, restart_vector))


def get_generic_cached_vectors(db_file="Cache/proximity_vectors.sqlite3", query_nodes=[2, 4], alpha=.25, cache_size=10):
    return io_utils.load_cached_vectors(db_file, query_nodes, alpha, cache_size)

def test_unnormalized_start_vector():

    cached_vectors = get_generic_cached_vectors()
    u = ppr.unnormalized_cached_start_vector(cached_vectors)
    print(u)

def test_total_sum_start_vector():

    cached_vectors = get_generic_cached_vectors(cache_size=1)
    u = ppr.total_sum_cached_start_vector(cached_vectors)
    print(u)
    print(sum(x for x in u.values()))

def test_twice_normalized_start_vector():

    cached_vectors = get_generic_cached_vectors(cache_size=1)
    u = ppr.twice_normalized_cached_start_vector(cached_vectors)
    print(u)
    print(sum(x for x in u.values()))

def test_build_cached_start_vector():

    dimension = 35000
    cached_vectors = get_generic_cached_vectors()

    start_vector = ppr.build_cached_start_vector(dimension, cached_vectors, norm_method="twice_normalized")
    print(start_vector)

    start_vector = ppr.build_cached_start_vector(dimension, cached_vectors, norm_method="total_sum")
    print(start_vector)

    start_vector = ppr.build_cached_start_vector(dimension, cached_vectors)
    print(start_vector)

