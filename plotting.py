import generate_results as gr
import itertools

db_file = "Cache/proximity_vectors.sqlite3"

def plot_all_cache_size():
    for alpha in [.01, .1, .25]:
        for query_size in [10, 50, 200]:
            plot_cache_size(alpha, query_size)

def plot_cache_size(alpha, query_size):
    cache_sizes = [10, 100, 1000]

def plot_alpha():
    return None

def plot_query_size():
    return None
