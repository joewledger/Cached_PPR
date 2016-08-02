import numpy as np
import itertools
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


def get_summary_dict(results_dict):
    return {key: (np.mean(value), np.std(value)) for key, value in results_dict.items()}


def get_field_values(summary_dict):
    methods, query_sizes, alphas, cache_sizes = set(), set(), set(), set()
    for method, query_size, alpha, cache_size in summary_dict.keys():
        methods.add(method)
        query_sizes.add(int(query_size))
        alphas.add(float(alpha))
        cache_sizes.add(int(cache_size))
    return methods, sorted(query_sizes), sorted(alphas), sorted(cache_sizes)


def plot_alpha(base_dir, summary_dict):
    base_file = base_dir + "plot_alpha"
    methods, query_sizes, alphas, cache_sizes = get_field_values(summary_dict)
    for query_size, cache_size in itertools.product(*[query_sizes, cache_sizes]):
        means_dict = {method: [] for method in methods}
        std_dev_dict = {method: [] for method in methods}
        for method, alpha in itertools.product(*[methods, alphas]):
            entries = get_matching_entries(summary_dict, method, query_size, alpha, cache_size)[0]
            means_dict[method].append(entries[1][0])
            std_dev_dict[method].append(entries[1][1])
        for value in means_dict.values():
            plt.plot(alphas, value)

        plt.xlabel("Alpha")
        plt.ylabel("Number of Iterations")
        plt.title("Alpha vs. Number of Iterations (|Q| = %d, Cache Size = %d)" % (query_size, cache_size))

        mi, ma = get_max_and_min(summary_dict)
        plt.ylim(mi, ma)

        save_file = "%s_query_size_%d_cache_size_%d.png" % (base_file, query_size, cache_size)
        plt.savefig(save_file)
        plt.close()


def plot_query_size(base_dir, summary_dict):
    base_file = base_dir + "plot_query_size"
    methods, query_sizes, alphas, cache_sizes = get_field_values(summary_dict)
    for alpha, cache_size in itertools.product(*[alphas, cache_sizes]):
        means_dict = {method: [] for method in methods}
        std_dev_dict = {method: [] for method in methods}
        for method, query_size in itertools.product(*[methods, query_sizes]):
            entries = get_matching_entries(summary_dict, method, query_size, alpha, cache_size)[0]
            means_dict[method].append(entries[1][0])
            std_dev_dict[method].append(entries[1][1])
        for value in means_dict.values():
            plt.plot(query_sizes, value)

        plt.xlabel("Query Size")
        plt.ylabel("Number of Iterations")
        plt.title("Query Sizes vs. Number of Iterations (Alpha = %.2f, Cache Size = %d)" % (alpha, cache_size))
        mi, ma = get_max_and_min(summary_dict)
        plt.ylim(mi, ma)

        save_file = "%s_alpha_%.2f_cache_size_%d.png" % (base_file, alpha, cache_size)
        plt.savefig(save_file)
        plt.close()


def plot_cache_size(base_dir, summary_dict):
    base_file = base_dir + "plot_cache_size"
    methods, query_sizes, alphas, cache_sizes = get_field_values(summary_dict)
    for alpha, query_size in itertools.product(*[alphas, query_sizes]):
        means_dict = {method: [] for method in methods}
        std_dev_dict = {method: [] for method in methods}
        for method, cache_size in itertools.product(*[methods, cache_sizes]):
            entries = get_matching_entries(summary_dict, method, query_size, alpha, cache_size)[0]
            means_dict[method].append(entries[1][0])
            std_dev_dict[method].append(entries[1][1])
        for key in means_dict.keys():
            plt.errorbar(cache_sizes, means_dict[key], yerr=std_dev_dict[key])

        plt.xlabel("Cache Size")
        plt.ylabel("Number of Iterations")
        plt.title("Cache Sizes vs. Number of Iterations (Alpha = %.2f, Query Size = %d)" % (alpha, query_size))

        mi, ma = get_max_and_min(summary_dict)
        plt.ylim(mi, ma)

        save_file = "%s_alpha_%.2f_query_size_%d.png" % (base_file, alpha, query_size)
        plt.savefig(save_file)
        plt.close()


def get_matching_entries(summary_dict, method, query_size, alpha, cache_size):
    items = []
    for item in summary_dict.items():
        key = item[0]
        if(key[0] == method and int(key[1]) == query_size and float(key[2]) == alpha and int(key[3]) == cache_size):
            items.append(item)
    return items


def get_max_and_min(summary_dict):
    iterable = [x[0] for x in summary_dict.values()]
    return min(iterable) - 5, max(iterable) + 5


if __name__ == "__main__":
    datafile = "Results/test.txt"
    save_dir = "Plots/Test/"

    results_dict = read_results_dict(datafile)
    summary_dict = get_summary_dict(results_dict)

    #plot_alpha(save_dir, summary_dict)
    #plot_query_size(save_dir, summary_dict)
    plot_cache_size(save_dir, summary_dict)
