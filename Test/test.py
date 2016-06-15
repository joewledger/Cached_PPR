import build_cache
import ppr
import pickle
import random

def test_build_cache():
    
    cache_name = "Email-Enron"
    alpha = .5
    path = "Cache/%s/%s/" % (cache_name, str(alpha))
    weight_matrix = ppr.read_csr_matrix("Data/%s.mat" % cache_name)
    cache_nodes = range(0,500)

    build_cache.build_cache(path,weight_matrix,alpha,cache_nodes=cache_nodes)


def test_cache_equivalence():

    alpha = .5
    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")

    vector = pickle.load(open("Cache/Email-Enron/0.5/3.p","rb"))
    ppr.print_top_k_indices_and_scores(vector,5)

    restart_vector = ppr.get_restart_vector(weight_matrix.shape[0],[3])
    vector2,iterations = ppr.ppr(weight_matrix, restart_vector.copy(), restart_vector, alpha)
    ppr.print_top_k_indices_and_scores(vector2,5)

def test_cached_ppr():

    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
    cache_path = "Cache/Email-Enron/0.5/"
    query_nodes = range(0,5)
    alpha = .5
    k = 5

    print(ppr.cached_ppr(weight_matrix,cache_path, query_nodes,alpha,k))

def test_cached_start_vector():

    dimension = 36692
    cache_path = "Cache/Email-Enron/0.5/"
    query_nodes = random.sample(range(0,500),5)
    cached_vectors = [pickle.load(open("%s%d.p" % (cache_path,i),"rb")) for i in query_nodes]
    k = 5

    ppr.get_cached_start_vector(dimension, cached_vectors,k)

def test_get_top_k_index_value_tuples():
    
    vector = pickle.load(open("Cache/Email-Enron/0.5/25.p","rb"))
    print(ppr.get_top_k_index_value_tuples(vector,10))

def test_cached_and_generic_ppr():

    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
    cache_path = "Cache/Email-Enron/0.5/"
    query_nodes = random.sample(range(0,500),100)
    alpha = .01
    k = 10000

    print(ppr.cached_ppr(weight_matrix,cache_path, query_nodes,alpha,k))

    print(ppr.generic_ppr(weight_matrix,query_nodes,alpha))
