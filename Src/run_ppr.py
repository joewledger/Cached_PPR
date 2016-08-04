import io_utils as io
import ppr
import start_vectors as sv
import vector_cache as vc

filename = "Data/Email-Enron.mat"

weight_matrix = io.load_csr_matrix(filename)
query_nodes = [2000]
alpha = .01


r1 = ppr.standard_ppr(weight_matrix, query_nodes, alpha)
io.print_vector_acsending(r1.final_vector, 20)
print(r1.num_iterations)
print("\n")

"""
vector_cache = vc.vector_cache()
vector_cache.build_cache(weight_matrix, [query_nodes], [alpha])
cache_size = 10

r2 = ppr.cached_ppr(weight_matrix, query_nodes, vector_cache, cache_size, alpha, norm_method=sv.twice_normalized)
v2 = ppr.trim_vector(r2.final_vector, 10)

print(v2)
print(r2.num_iterations)
"""

r3 = ppr.standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.chebyshev_ppr)
io.print_vector_acsending(r3.final_vector, 20)
print(r3.num_iterations)
