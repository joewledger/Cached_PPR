import os
import ppr
import pickle

def build_cache(path, weight_matrix, alpha,num_threads=5,cache_nodes=None):
    dimension = weight_matrix.shape[0]
    os.makedirs(path,exist_ok=True)
    if(not cache_nodes):
        cache_nodes = range(0,dimension)


    def pickle_node_i(i):
        start_vector = ppr.get_start_vector(dimension,[i])
        restart_vector = start_vector.copy()

        final_vector,iterations = ppr.ppr(weight_matrix,start_vector,restart_vector,alpha)
        full_path = "%s%d.p" % (path,i)
        pickle.dump(final_vector,open(full_path,"wb"))

    for i in cache_nodes:
        pickle_node_i(i)