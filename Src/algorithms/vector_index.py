import ppr_interface
import vector_utils as vu
import itertools
import pickle
from multiprocessing import Manager, Pool


class Vector_Index:

    def __init__(self):
        self.index = {}
        self.trimmed_index = {}

    def build_index(self, weight_matrix, query_sets, alphas, eps=1E-10, num_processes=1):
        if(type(query_sets[0]) == list):
            #flattens the list of lists into a single set
            all_query_nodes = set(itertools.chain(*query_sets))
        else:
            all_query_nodes = query_sets

        #Namespace and manager object are for creating the index with multiple proccesses.
        manager = Manager()
        namespace = manager.Namespace()
        namespace.weight_matrix = weight_matrix
        namespace.eps = eps

        parameters = [(namespace, query_node, alpha) for query_node, alpha in itertools.product(all_query_nodes, alphas)]

        #Creates the index in parallel by mapping the parameter sets to the get_proximity_vector function
        pool = Pool(processes=num_processes)
        result = pool.starmap(get_proximity_vector, parameters)

        #Saves the proximity_vectors
        for query_node, alpha, vector in result:
            self.insert_vector(query_node, alpha, vector)

    #Gets the indexed vector for the given node_id and alpha
    def get_vector(self, node_id, alpha):
        encoded_alpha = self.encode_alpha(alpha)
        return self.index[(node_id, encoded_alpha)]

    #Gets the trimmed vector from the index
    #If the index contains the full vector for the given node_id and alpha but not the trimmed vector
    #We trim the vector and store the trimmed_vector in the index
    def get_trimmed_vector(self, node_id, alpha, index_size):
        encoded_alpha = self.encode_alpha(alpha)
        trimmed_key = (node_id, encoded_alpha, index_size)
        standard_key = (node_id, encoded_alpha)

        #Get trimmed_vector from index if it is in the index
        if(trimmed_key in self.trimmed_index.keys()):
            return self.trimmed_index[trimmed_key]
        #Otherwise trim the full vector, then store and return the trimmed vector
        elif(standard_key in self.index.keys()):
            trimmed_vector = vu.trim_vector(self.index[standard_key], index_size)
            self.trimmed_index[trimmed_key] = trimmed_vector
            return trimmed_vector

    #Inserts a vector into the index for a given node_id and alpha
    def insert_vector(self, node_id, alpha, vector):
        encoded_alpha = self.encode_alpha(alpha)
        self.index[(node_id, encoded_alpha)] = vector

    #Encodes alpha as a string
    #This is so that alpha (a floating point number) can be used as a dictionary key
    #Simply takes the first three points after the decimal point
    def encode_alpha(self, alpha):
        return "%.3f" % alpha

    #Gets a list of indexed vectors for a set of query nodes
    def get_vector_list(self, query_set, alpha, index_size=None):
        if(index_size):
            return [self.get_trimmed_vector(node_id, alpha, index_size) for node_id in query_set]
        else:
            return [self.get_vector(node_id, alpha) for node_id in query_set]

    #Saves the dictionary of full proximity vectors to a binary file
    def save_to_file(self, filename):
        pickle.dump(self.index, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    #Loads a dictionary of full proximity vectors from a file
    def load_from_file(self, filename):
        self.index = pickle.load(open(filename, 'rb'))


def get_proximity_vector(namespace, query_node, alpha):
    return (query_node, alpha, ppr_interface.get_proximity_vector(namespace.weight_matrix, query_node, alpha, eps=namespace.eps))
