
import ppr
import scipy as sp

def test_scalar_multiply():
	weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")

	query_nodes, start_vector = ppr.generate_random_start_vector(weight_matrix.shape[0], 5)
	restart_vector = start_vector.copy()
	curr_vector, iterations = ppr.ppr(weight_matrix, start_vector, restart_vector, .01)

	scalar = 2

	start_vector2 = start_vector.multiply(scalar)
	curr_vector2, iterations2 = ppr.ppr(weight_matrix, start_vector2, restart_vector, .01)

	start_vector3 = start_vector.multiply(scalar)
	restart_vector3 = restart_vector.multiply(scalar)
	curr_vector3, iterations3 = ppr.ppr(weight_matrix, start_vector3, restart_vector3, .01)

	print(curr_vector.sum(0))
	print(curr_vector2.sum(0))
	print(curr_vector3.sum(0))

	print(iterations, iterations2, iterations3)

def test_l1_norm():

	weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
	query_nodes, start_vector = ppr.generate_random_start_vector(weight_matrix.shape[0], 5)
	print(start_vector)
	print(start_vector.sum(0))
	#print(sp.linalg.norm(start_vector,ord=0,axis=0))

test_scalar_multiply()