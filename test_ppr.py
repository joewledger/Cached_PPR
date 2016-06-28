
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
	print(e1)


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
