import io_utils


def test_load_cached_vectors():

    db_file = "Cache/proximity_vectors.sqlite3"
    query_nodes = [2, 4]
    alpha = .25
    cache_size = 10

    cached_vectors = io_utils.load_cached_vectors(db_file, query_nodes, alpha, cache_size)
    print(cached_vectors)


def test_get_num_iterations():

    db_file = "Cache/proximity_vectors.sqlite3"
    kwargs = dict(alpha=.01, query_size=10, cache_size=100, norm_method="Unnormalized")

    i = io_utils.get_num_iterations(db_file, **kwargs)
    print(i)
