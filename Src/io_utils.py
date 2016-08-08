import scipy.io as spio
import numpy as np
import pickle


def load_query_nodes(filename):
	return pickle.load(open(filename, "rb"))


def load_csr_matrix(filename):
    if(filename[-4:] == ".mat"):
        return spio.loadmat(filename)["normalizedNetwork"]
    elif(filename[-4:] == ".mtx"):
        return spio.mmread(filename)


def print_vector_acsending(vector, k):
    data = vector.toarray().flatten()
    ind = np.argpartition(data, -k)[-k:]
    ind = sorted(ind, key=lambda i: data[i])
    output = "\n".join("(%d, 0)\t%.8f" % (i, data[i]) for i in ind)
    print(output)
