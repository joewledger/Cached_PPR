import ppr
import random
import pickle
import matplotlib.pyplot as plt
import os.path
import numpy as np
import itertools

def save_cache_size_effect():

    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    num_permutations = 10
    alpha = .01
    query_size = 100
    cache_size = [1,10,20,50,100,200,500,1000]
    cache_path = "Cache/Email-Enron/0.5/"
    
    permuted_query_nodes = [random.sample(range(0,dimension),query_size) for p in range(0,num_permutations)]

    cached_score = lambda norm_method,cache_size,query_nodes : ppr.cached_ppr(weight_matrix,cache_path, query_nodes
                                                                               ,alpha,cache_size,norm_method=norm_method)[1]

    cached_permuted = lambda norm_method,cache_size : [cached_score(norm_method,cache_size,p) for p in permuted_query_nodes]

    generic_scores = [ppr.generic_ppr(weight_matrix,p,alpha)[1] for p in permuted_query_nodes]
    #Map cache_size to a list of iteration scores
    total_sum_scores = {cs : cached_permuted("total_sum",cs) for cs in cache_size}
    num_queries_scores = {cs : cached_permuted("num_queries",cs) for cs in cache_size}
    unnormalized_scores = {cs : cached_permuted(None,cs) for cs in cache_size}

    pickle.dump(cache_size,open("Results/cache_size.p","wb"))
    pickle.dump(generic_scores, open("Results/cache_size_generic.p","wb"))
    pickle.dump(total_sum_scores, open("Results/cache_size_total_sum.p","wb"))
    pickle.dump(num_queries_scores, open("Results/cache_size_num_queries.p","wb"))
    pickle.dump(unnormalized_scores, open("Results/cache_size_unnormalized.p","wb"))

def plot_cache_size_effect():

    cache_size = pickle.load(open("Results/cache_size.p","rb"))
    generic_scores = pickle.load(open("Results/cache_size_generic.p","rb"))
    total_sum_scores = pickle.load(open("Results/cache_size_total_sum.p","rb"))
    num_queries_scores = pickle.load(open("Results/cache_size_num_queries.p","rb"))
    unnormalized_scores = pickle.load(open("Results/cache_size_unnormalized.p","rb"))

    to_list = lambda d : [d[x] for x in cache_size]
    apply_function = lambda d , func : [func(x) for x in to_list(d)]
    apply_mean_std = lambda d : (apply_function(d,np.mean), apply_function(d,np.std))
    copy = lambda value , n : [value for i in range(0,n)]

    g = copy(np.mean(generic_scores),len(cache_size)),copy(np.std(generic_scores),len(cache_size))
    t = apply_mean_std(total_sum_scores)
    n = apply_mean_std(num_queries_scores)
    u = apply_mean_std(unnormalized_scores)

    arr = [g,t,n,u]
    cache_plot(cache_size,arr,4,"Results/cache_size_full.png")
    cache_plot(cache_size,arr,3,"Results/cache_size.png")
    
def cache_plot(cache_size,arr,num_fields,savefile):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    legend_labels = ["Generic","Total Sum","Number of Queries", "Unnormalized"]

    for i,m in enumerate(arr[:num_fields]):
        axes.errorbar(cache_size,m[0],yerr=m[1],label=legend_labels[i])

    axes.set_xscale("log")
    axes.set_xlabel("Cache Size")
    axes.set_ylabel("Number of Iterations")
    axes.set_title(r"Cache Size vs. Number of Iterations ($\alpha$=.01,|Q|=100)")
    axes.legend(loc=5)
    plt.savefig(savefile)

def save_alpha_effect():

    weight_matrix = ppr.read_csr_matrix("Data/Email-Enron.mat")
    dimension = weight_matrix.shape[0]
    num_permutations = 10
    query_sizes = [5,10,50,100]
    alphas = [.1 * x for x in range(1,11)]
    cache_size = 10
    cache_path = "Cache/Email-Enron/0.5/"
    savepath = "Results/Alpha/"
    os.makedirs(savepath,exist_ok=True)

    combinations = list(itertools.product(query_sizes,alphas))

    for c in combinations:
        query_size = c[0]
        alpha = c[1]

        generic_scores = []
        sum_scores = []
        num_query_scores = []

        for i in range(0,num_permutations):
            
            query_nodes = random.sample(range(0,500),query_size)
            generic_scores.append(ppr.generic_ppr(weight_matrix,query_nodes,alpha)[1])
            sum_scores.append(ppr.cached_ppr(weight_matrix,cache_path, query_nodes,alpha,cache_size,norm_method="total_sum")[1])
            num_query_scores.append(ppr.cached_ppr(weight_matrix,cache_path, query_nodes,alpha,cache_size,norm_method="num_queries")[1])

        pickle.dump(generic_scores,open("Results/Alpha/%s_%.1f_generic.p" % (query_size,alpha), "wb"))
        pickle.dump(sum_scores,open("Results/Alpha/%s_%.1f_sum_scores.p" % (query_size,alpha), "wb"))
        pickle.dump(num_query_scores,open("Results/Alpha/%s_%.1f_num_queries.p" % (query_size,alpha), "wb"))



def plot_alpha_effect():
    return None


if __name__ == "__main__":
    if(not os.path.isfile("Results/cache_size_generic.p")):
        save_cache_size_effect()
    if(not os.path.isfile("Results/cache_size.png")):
        plot_cache_size_effect()

    if(not os.path.isdir("Results/Alpha/")):
        save_alpha_effect()
    if(not os.path.isdir("Results/Alpha/")):
        plot_alpha_effect()

