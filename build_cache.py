import os
import ppr
import pickle
from joblib import Parallel, delayed
import multiprocessing

def build_cache(path, weight_matrix, alpha,num_cores=5,cache_nodes=None):
    os.makedirs(path,exist_ok=True)
    if(not cache_nodes):
        cache_nodes = range(0,weight_matrix.shape[0])

    Parallel(n_jobs=num_cores)(delayed(pickle_node_i)(path,weight_matrix,alpha,i) for i in cache_nodes)

def pickle_node_i(path,weight_matrix,alpha,i):
    start_vector = ppr.get_start_vector(weight_matrix.shape[0],[i])
    restart_vector = start_vector.copy()

    final_vector,iterations = ppr.ppr(weight_matrix,start_vector,restart_vector,alpha)
    full_path = "%s%d.p" % (path,i)
    pickle.dump(final_vector,open(full_path,"wb"))

if __name__ == "__main__":

    cache_name = "Email-Enron"
    alpha = .01
    path = "Cache/%s/%s/" % (cache_name, str(alpha))
    weight_matrix = ppr.read_csr_matrix("Data/%s.mat" % cache_name)
    build_cache(path,weight_matrix,alpha,num_cores=10)