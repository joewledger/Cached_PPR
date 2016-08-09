import vector_cache
import io_utils as io


def test_build_cache():
    weight_matrix = io.load_csr_matrix("Data/test1000.mtx")
    cache = vector_cache.vector_cache()
    cache.build_cache(weight_matrix, [x for x in range(6, 107)], [.25, .5], num_processes=4)
    print(cache.get_vector(69, .5))
