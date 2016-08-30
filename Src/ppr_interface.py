import vector_utils as vu
import ppr


def standard_ppr(weight_matrix, query_nodes, alpha, ppr_method=ppr.standard_ppr):
    restart_vector = vu.get_restart_vector(weight_matrix.shape[0], query_nodes)
    start_vector = restart_vector.copy()
    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)


def indexed_ppr(weight_matrix, query_nodes, vector_cache, cache_size, alpha, ppr_method=ppr.standard_ppr, norm_method=vu.unnormalized):
    dimension = weight_matrix.shape[0]

    vector_list = vector_cache.get_vector_list(query_nodes, alpha, cache_size=cache_size)
    start_vector = norm_method(vector_list)
    restart_vector = vu.get_restart_vector(dimension, query_nodes)

    return ppr_method(weight_matrix, start_vector, restart_vector, alpha)
