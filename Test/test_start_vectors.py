import io_utils as io
import generate_results as gr
import start_vectors as sv


def test_start_vectors():
    weight_matrix = io.load_csr_matrix("Data/test.mtx")
    query_sets = [[5, 37, 84]]
    alphas = [.01]

    cache = gr.generate_vector_cache(weight_matrix, query_sets, alphas)
    vector_list = cache.get_vector_list(query_sets[0], alphas[0], cache_size=10)

    u = sv.unnormalized_start_vector(vector_list)
    ts = sv.total_sum_vector(vector_list)
    tn = sv.twice_normalized_vector(vector_list)

    print(u)
    print(ts)
    print(tn)

    print(u.sum(), ts.sum(), tn.sum())
