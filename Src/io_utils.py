import scipy.io as spio
import sqlite3
import random
import string
import pickle
import os


def load_csr_matrix(filename):
    if(filename[-4:] == ".mat"):
        return spio.loadmat(filename)["normalizedNetwork"]
    elif(filename[-4:] == ".mtx"):
        return spio.mmread(filename)


def load_cached_vectors(vector_filepaths, cache_size):
    raise NotImplementedError


def pickle_proximity_vector(proximity_filepath, proximity_vector):
    try:
        os.makedirs(proximity_filepath[:proximity_filepath.rfind("/")])
    except:
        pass
    pickle.dump(proximity_vector, open(proximity_filepath, "wb"))


class DBWrapper:

    def __init__(self, db_file, tables_file="Src/initialize.sql"):
        self.db_file = db_file
        self.tables_file = tables_file
        self.has_open_connection = False

    def open_connection(self):
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()
        self.has_open_connection = True

    def close_connection(self):
        self.connection.close()
        self.has_open_connection = False

    def check_connection(self):
        if(not self.has_open_connection):
            raise Exception("No open database connection")

    def initialize_all_tables(self):
        self.check_connection()
        file_handle = open(self.tables_file, "rb")
        for line in file_handle.readlines():
            command = line.decode('utf-8').strip()
            try:
                self.cursor.execute(command)
            except:
                print("Failed: " + command)

    #Returns a python dictionary mapping {alpha_id -> alpha}
    def get_alpha_mapping(self):
        raise NotImplementedError

    def get_alpha_mapping_and_update_table(self, alphas, threshold=.001):
        raise NotImplementedError

    def get_closest_alpha_tuple(self, alpha):
        raise NotImplementedError

    def get_closest_alpha_tuple_and_update_table(self, alpha, threshold=.001):
        self.cursor.execute('SELECT * FROM alpha_ids ORDER BY ABS( alpha - %s) ASC LIMIT 1' % str(alpha))
        closest_tuple = self.cursor.fetchone()
        if(not closest_tuple):
            new_tuple = (0, alpha)
            self.cursor.execute('INSERT INTO alpha_ids VALUES (%d, %f)' % new_tuple)
            self.connection.commit()
            return new_tuple
        if(abs(alpha - closest_tuple[1]) < threshold):
            return closest_tuple
        else:
            self.cursor.execute('SELECT alpha_id FROM alpha_ids ORDER BY alpha_id desc LIMIT 1')
            alpha_id, = self.cursor.fetchone()
            alpha_id += 1
            self.cursor.execute('INSERT INTO alpha_ids VALUES (%d, %f)' % (alpha_id, alpha))
            self.connection.commit()
            return alpha_id, alpha

    def get_proximity_vector_filepaths(self):
        raise NotImplementedError

    def update_proximity_vector_filepaths(self, network_filepath, query_node_to_proximity_vector_filepath, alpha_id, vector_length):
        query = "INSERT OR IGNORE INTO proximity_vectors VALUES (?,?,?,?,?)"

        proximity_filepath = lambda first_node: query_node_to_proximity_vector_filepath[first_node]
        nodes = list(query_node_to_proximity_vector_filepath.keys())

        entries = [(network_filepath, first_node, alpha_id, proximity_filepath(first_node), vector_length) for first_node in nodes]

        print(entries)

        self.cursor.executemany(query, entries)
        self.connection.commit()

    def get_unique_proximity_filepaths(self, path_prefix="Cache/", path_suffix=".pickle", count=1):
        self.check_connection()

        def filepath_gen():
            while(True):
                rand_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
                yield path_prefix + rand_str + path_suffix

        query = 'SELECT DISTINCT proximity_filepath FROM proximity_vectors'
        self.cursor.execute(query)
        existing_filepaths = self.cursor.fetchall()

        fg = filepath_gen()
        unique_filepaths = []
        while(len(unique_filepaths) < count):
            path = next(fg)
            if(path not in existing_filepaths):
                unique_filepaths.append(path)
                existing_filepaths.append(path)
        return unique_filepaths

    def get_results(self, network_filepath, query_ids):
        raise NotImplementedError

    def get_method_id(self, method_name):
        raise NotImplementedError

    def update_results(self, param_to_result_mapping):
        raise NotImplementedError

    def get_most_recent_queries(self, network_filepath, count=10):
        raise NotImplementedError

    def update_query_sets(self, network_filepath, query_sets):
        raise NotImplementedError

    def get_query_set(self, query_id):
        raise NotImplementedError
