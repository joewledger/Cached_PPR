import algorithms.ppr as ppr
import algorithms.io_utils as io
import algorithms.vector_utils as vu

"""
Code for demonstrating that the L1-norm of the restart vector will be the same as the L1-norm of the final converged vector
"""

weight_matrix = io.load_csr_matrix("Data/Email-Enron.mat")
dimension = weight_matrix.shape[0]
query_set = [10, 20, 30]
alpha = .5

restart_vector = vu.get_restart_vector(dimension, query_set)
start_vector = restart_vector.copy()

result = ppr.standard_ppr(weight_matrix, start_vector, restart_vector, alpha)

print(vu.get_l1_norm(result.final_vector))

result2 = ppr.standard_ppr(weight_matrix, start_vector, restart_vector * alpha, alpha)

print(vu.get_l1_norm(result2.final_vector))
