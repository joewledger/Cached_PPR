
import ppr
import scipy as sp


def test_scalar_multiply():
	weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
	query_nodes, start_vector = ppr.generate_random_start_vector(weight_matrix.shape[0], 5)
	restart_vector = start_vector.copy()
	alpha = .01

	def scalar_experiment(weight_matrix, alpha, start_vector,restart_vector, c, d):
		return ppr.ppr(weight_matrix, start_vector.multiply(c), restart_vector.multiply(d), alpha)

	v1, i1 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,1.0,1.0)
	v2, i2 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,5.0,1.0)
	v3, i3 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,1.0,5.0)
	v4, i4 = scalar_experiment(weight_matrix,alpha,start_vector,restart_vector,5.0,5.0)

	print(i1,i2,i3,i4)



def test_l1_norm():

	weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
	query_nodes, start_vector = ppr.generate_random_start_vector(weight_matrix.shape[0], 5)
	print(start_vector)
	print(start_vector.sum(0))
