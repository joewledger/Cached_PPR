import algorithms.vector_utils as vu
import algorithms.ppr as ppr


"""
Methods in this module provide an interface for using the PPR query and top-K algorithms using a
set of query nodes instead of a start vector

#This also allows for easy use of the vector index.
"""


#Interface for using the non-indexed PPR algorithms with a set of query nodes
#Default PPR method is the standard implementation
#Chebyshev is also possible, change ppr_method to ppr.chebyshev_ppr
def standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.standard_ppr):
    restart_vector = vu.get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


#Interface for using the indexed PPR algorithms with a set of query nodes
#Default PPR method is the standard implementation
#Chebyshev is also possible, change ppr_method to ppr.chebyshev_ppr
#Must provide a vector index, an index size, and can change the normalization method
#Possible normalization methods are vu.total_sum, and vu.twice_normalized
def indexed_ppr(weight_matrix, query_nodes, vector_index, index_size, alpha, ppr_method=ppr.standard_ppr, norm_method=vu.twice_normalized):
    dimension = weight_matrix.shape[0]

    vector_list = vector_index.get_vector_list(query_nodes, alpha, index_size=index_size)
    start_vector = norm_method(vector_list)
    restart_vector = vu.get_restart_vector(dimension, query_nodes)

    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


def standard_top_k(weight_matrix, query_nodes, alpha, k, top_k_method=ppr.chebyshev_top_k):
    restart_vector = vu.get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return top_k_method(weight_matrix, start_vector, restart_vector, alpha, k)


def indexed_top_k(weight_matrix, query_nodes, vector_index, index_size, alpha, k,
                  top_k_method=ppr.chebyshev_top_k, norm_method=vu.twice_normalized, **kwargs):

    dimension = weight_matrix.shape[0]

    vector_list = vector_index.get_vector_list(query_nodes, alpha, index_size=index_size)
    start_vector = norm_method(vector_list)
    restart_vector = vu.get_restart_vector(dimension, query_nodes)

    return top_k_method(weight_matrix, start_vector, restart_vector, alpha, k, **kwargs)


#Gets the proximity vector for a given query node using the specified PPR method
#Default PPR method is chebyshev, but standard PPR will work the same (but slower)
def get_proximity_vector(weight_matrix, query_node, alpha, ppr_method=ppr.chebyshev_ppr):
    return standard_ppr(weight_matrix, [query_node], alpha, ppr_method=ppr_method).final_vector
