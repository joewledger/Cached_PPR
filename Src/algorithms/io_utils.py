import scipy.io as spio
import pickle


def get_index_filepath(network_filepath):
    return network_filepath.replace("Data/", "Index/")[:-4] + ".pickle"


def get_indexed_queries_filepath(network_filepath):
    return network_filepath.replace("Data/", "Index/")[:-4] + "-queries.pickle"


def load_query_nodes(filename):
    return pickle.load(open(filename, "rb"))


#Loads a CSR matrix in either .mat or .mtx format
def load_csr_matrix(filename):
    if(filename[-4:] == ".mat"):
        return spio.loadmat(filename)["normalizedNetwork"]
    elif(filename[-4:] == ".mtx"):
        return spio.mmread(filename)
