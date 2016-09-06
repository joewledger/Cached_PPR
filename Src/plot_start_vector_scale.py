import algorithms.ppr
import io_utils as io
import vector_utils as vu
import itertools
import matplotlib.pyplot as plt
import numpy as np


def main(input_file, save_file, alpha, query_size, num_permutations):

    weight_matrix = io.load_csr_matrix(input_file)
    dimension = weight_matrix.shape[0]
    query_sets = vu.get_query_sets(num_permutations, query_size, range(dimension))
    scale_factors = [1.0, 5.0]
    scale_factors_combinations = list(itertools.product(*[scale_factors, scale_factors]))
    results = {(c, d): [] for c, d in scale_factors_combinations}
    for q in query_sets:
        restart_vector = ppr.get_restart_vector(dimension, q)
        start_vector = restart_vector.copy()
        scaled_ppr = lambda c, d: ppr.ppr(weight_matrix, start_vector * c, restart_vector * d, alpha).num_iterations
        for c, d in results.keys():
            results[(c, d)].append(scaled_ppr(c, d))

    print(results)
    means = [np.mean(results[s]) for s in scale_factors_combinations]
    stds = [np.std(results[s]) for s in scale_factors_combinations]
    ind = np.arange(0, len(means), 1)
    width = .35
    fig, ax = plt.subplots()
    ax.bar(ind + width, means, width, yerr=stds, ecolor='k')
    ax.set_title(r"Effect of changing start and restart vector scales ($\alpha$=%.2f, |Q|=%d)" % (alpha, query_size))
    ax.set_xlabel("Scaling factor combination")
    xticks = ["c=%d, d=%d" % (int(c), int(d)) for c, d in scale_factors_combinations]
    plt.xticks(np.arange(.5, 4, 1), xticks)
    ax.set_ylabel("Average number of iterations")
    ax.set_ylim((0, max(means) * 1.2))
    plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":
    input_file = "Data/Email-Enron.mat"
    save_file = "Plots/start_vector_scale.png"
    alpha = 0.01
    query_size = 200
    num_permutations = 10
    main(input_file, save_file, alpha, query_size, num_permutations)
