import build_cache
import ppr
import pickle

def test_build_cache():
    
    cache_name = "Email-Enron"
    alpha = .5
    path = "Cache/%s/%s/" % (cache_name, str(alpha))
    weight_matrix = ppr.read_csr_matrix("Data/%s.mat" % cache_name)
    cache_nodes = range(0,5)

    build_cache.build_cache(path,weight_matrix,alpha,cache_nodes=cache_nodes)


def test_cache_equivalence():

    alpha = .5
    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")

    vector = pickle.load(open("Cache/Email-Enron/0.5/3.p","rb"))
    ppr.print_top_k_indices_and_scores(vector,5)

    start_vector = ppr.get_start_vector(weight_matrix.shape[0],[3])
    vector2,iterations = ppr.ppr(weight_matrix, start_vector, start_vector.copy(), alpha)
    ppr.print_top_k_indices_and_scores(vector2,5)