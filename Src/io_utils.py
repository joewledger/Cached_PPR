import scipy.io as spio


def load_csr_matrix(filename):
    if(filename[-4:] == ".mat"):
        return spio.loadmat(filename)["normalizedNetwork"]
    elif(filename[-4:] == ".mtx"):
        return spio.mmread(filename)
