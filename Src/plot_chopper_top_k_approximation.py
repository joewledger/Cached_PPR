import algorithms.io_utils as io
import algorithms.vector_index as vector_index
import algorithms.ppr_interface as ppr_interface
import algorithms.vector_utils as vu
import matplotlib.pyplot as plt
import random


def plot(weight_matrix, query_nodes, alpha, index, index_size, k_values):

    chopper_results = [ppr_interface.standard_top_k(weight_matrix, query_nodes, alpha, k) for k in k_values]
    indexed_chopper_results = [ppr_interface.standard_top_k(weight_matrix, query_nodes, alpha, k) for k in k_values]
    chopper_multiply_results = [ppr_interface.standard_top_k(weight_matrix, query_nodes, alpha, k, adjust_error=True) for k in k_values]
    indexed_chopper_multiply_results = [ppr_interface.indexed_top_k(weight_matrix, query_nodes, index, index_size, alpha, k, adjust_error=True) for k in k_values]

    cm_recall = [vu.measure_top_k_recall(chopper_results[i].final_vector, chopper_multiply_results[i].final_vector, k_values[i]) for i in range(len(k_values))]
    icm_recall = [vu.measure_top_k_recall(chopper_results[i].final_vector, indexed_chopper_multiply_results[i].final_vector, k_values[i]) for i in range(len(k_values))]

    plt.plot(k_values, cm_recall, label="CHOPPER-MULTIPLY")
    plt.plot(k_values, icm_recall, label="Indexed CHOPPER-MULTIPLY")
    plt.xlabel("k")
    plt.ylabel("Recall")
    plt.title("Effect of k on Recall (Index Size %d)" % index_size)
    plt.legend()
    plt.savefig("Plots/Chopper_Multiply_Recall_%d.png" % index_size)
    plt.close()

    c_iter = [c.num_iterations for c in chopper_results]
    ic_iter = [ic.num_iterations for ic in chopper_results]
    cm_iter = [cm.num_iterations for cm in chopper_multiply_results]
    icm_iter = [icm.num_iterations for icm in indexed_chopper_multiply_results]

    plt.plot(k_values, c_iter, label="CHOPPER")
    plt.plot(k_values, ic_iter, label="Indexed CHOPPER")
    plt.plot(k_values, cm_iter, label="CHOPPER-MULTIPLY")
    plt.plot(k_values, icm_iter, label="Indexed CHOPPER-MULTIPLY")
    plt.xlabel("k")
    plt.ylabel("Number of iterations")
    plt.title("Effect of k on number of iterations (Index Size %d)" % index_size)
    plt.legend(loc=2)
    plt.savefig("Plots/Chopper_Multiply_Iterations_%d.png" % index_size)
    plt.close()


if __name__ == "__main__":
    weight_matrix = io.load_csr_matrix("Data/Email-Enron.mat")
    query_size = 10
    query_nodes = [x for x in random.sample(range(weight_matrix.shape[0]), query_size)]
    alpha = .01
    index_size = 10
    k_values = range(100, 1100, 100)
    index = vector_index.Vector_Index()
    index.build_index(weight_matrix, [query_nodes], [alpha])
    for index_size in [10, 100, 1000]:
        plot(weight_matrix, query_nodes, alpha, index, index_size, k_values)
