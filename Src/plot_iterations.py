import numpy as np
#import itertools
import matplotlib.pyplot as plt


def read_results_dict(filename):
    results_dict = {}

    reader = open(filename, "r")
    for line in reader.readlines():
        fields = line.strip().split("\t")
        key = tuple(fields[:4])
        value = int(fields[4])
        if(key in results_dict):
            results_dict[key].append(value)
        else:
            results_dict[key] = [value]
    reader.close()
    return results_dict


def generic_plot_all(results_dict, ind_variable_loc):
    #nrv stands for non relavant variables
    nrv = {1, 2, 3} - {ind_variable_loc}
    #nrvc stands for non relavant value combinations
    nrvc = {tuple([key[x] for x in nrv]) for key in results_dict.keys()}
    for nrv_combo in nrvc:
        plot(results_dict, ind_variable_loc, nrv, nrv_combo)


def plot(results_dict, ind_variable_loc, nrv_locs, nrv_combo):
    base_save_dir = "Plots/Iterations/"

    long_names = ["Method", "Query Size", "Alpha", "Index Size", "Iterations"]
    short_names = ["Method", "|Q|", r'$\alpha$', "|I|", "Iterations"]
    filehandle_names = ["", "query_size", "alpha", "index_size", ""]
    typecasts = [str, int, float, int, int]
    method_names = {"standard": "Standard", "twice_normalized": "Twice Normalized", "total_sum": "Total Sum"}
    ind_variable_name = long_names[ind_variable_loc]
    ind_filehandle_name = filehandle_names[ind_variable_loc]
    title_names = [short_names[l] for l in nrv_locs]
    filehandle_names = [filehandle_names[l] for l in nrv_locs]

    key_matches = lambda key: all(key[x] == nrv_combo[i] for i, x in enumerate(nrv_locs))
    get_typecasted_ind_value = lambda key: typecasts[ind_variable_loc](key[ind_variable_loc])

    relavant_keys = [key for key in results_dict.keys() if key_matches(key)]
    methods = ["standard", "twice_normalized", "total_sum"]
    ind_variable_values = sorted({get_typecasted_ind_value(x) for x in relavant_keys})
    for method in methods:
        matching_tuples = []
        for value in ind_variable_values:
            relevant_key = [k for k in relavant_keys if k[0] == method and get_typecasted_ind_value(k) == value][0]
            matching_tuples.append(relevant_key)
        means = [np.mean([int(x) for x in results_dict[t]]) for t in matching_tuples]
        stds = [np.std([int(x) for x in results_dict[t]]) for t in matching_tuples]
        plt.errorbar(ind_variable_values, means, yerr=stds, label=method_names[method])

    title = "%s vs Number of Iterations (%s=%s, %s=%s)" % (ind_variable_name, title_names[0], nrv_combo[0], title_names[1], nrv_combo[1])
    savefile = "%s%s_plot_%s_%s_%s_%s.png" % (base_save_dir, ind_filehandle_name, filehandle_names[0], nrv_combo[0], filehandle_names[1], nrv_combo[1])

    if(ind_variable_loc == 3):
        plt.xscale('log')

    plt.title(title)
    plt.legend()
    plt.savefig(savefile)
    plt.close()


if __name__ == "__main__":
    datafile = "Results/out.txt"
    results_dict = read_results_dict(datafile)
    generic_plot_all(results_dict, 1)
    generic_plot_all(results_dict, 2)
    generic_plot_all(results_dict, 3)
