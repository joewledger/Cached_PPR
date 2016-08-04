import io_utils as io
import ppr


filename = "Data/Email-Enron.mat"

weight_matrix = io.load_csr_matrix(filename)
query_nodes = [1001, 20034, 31056]
alpha = .01


result = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
final_vector = ppr.trim_vector(result.final_vector, 10)


print(result.num_iterations)
print(final_vector)
