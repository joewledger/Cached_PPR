import generate_results as gr
import itertools
import matplotlib as plt

db_file = "Cache/proximity_vectors.sqlite3"
base_out = "Results/"

def generic_line_plot(xlabel="Cache Size", ylabel= "Number of Iterations", title = "")


def make_all_plots():
	plot_alphas()
	plot_cache_sizes()
	plot_query_sizes()

