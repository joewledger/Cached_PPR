import ppr
from joblib import Parallel, delayed
import sqlite3
import itertools
import argparse


def build_cache(db_file, matrix_file, network_name, alphas, num_threads=5, top_k=200, max_dim=None):

    initialize_database(db_file, define_schemas())
    weight_matrix = ppr.read_csr_matrix(matrix_file)

    #dictionary_mapping alpha_ids to real floating point alpha values
    alpha_id_mapping = update_alpha_ids(db_file, alphas)

    dimension = weight_matrix.shape[0]
    if(max_dim):
        dimension = min(dimension, max_dim)
    #Cartesian product of node number and alpha_id
    #Should look something like this [(0,0), (0,1), (0,2), ... , (679, 0), (679, 1), ...]
    node_number_alpha_product = itertools.product(range(0, dimension), alpha_id_mapping.keys())

    parallel_params = lambda node_id, alpha_id: [db_file, weight_matrix, network_name, node_id, alpha_id, alpha_id_mapping[alpha_id], top_k]

    Parallel(n_jobs=num_threads)(delayed(save_proximity_vector)(*parallel_params(node_id, alpha_id))
                                 for node_id, alpha_id in node_number_alpha_product)


def save_proximity_vector(db_file, weight_matrix, network_name, node_id, alpha_id, alpha_value, top_k):
    proximity_vector, iterations = ppr.generic_ppr(weight_matrix, [node_id], alpha_value)

    index_value_tuples = ppr.get_top_k_index_value_tuples(proximity_vector, top_k)

    insertions = []

    for ranking, index_value in enumerate(reversed(index_value_tuples)):
        second_node_id, score = index_value
        insertions.append((network_name, node_id, second_node_id, ranking, alpha_id, score))

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.executemany('INSERT OR IGNORE INTO proximity_vectors VALUES (?,?,?,?,?,?)', insertions)
    conn.commit()
    conn.close()


#Gets a dictionary mapping of alpha_id -> alpha, consistent with the existing database
#If there is a new value of alpha not within .0001 of any alpha in the database, make
#a new entry in the database for it and return the new key-value pairing in the dictionary
def update_alpha_ids(db_file, alphas):
    mapping = {}
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for a in alphas:
        c.execute('SELECT * FROM alpha_ids')
        if(not c.fetchall()):
            new_id = 0
            mapping[new_id] = a
            c.execute('INSERT INTO alpha_ids VALUES (%d,%s)' % (new_id, a))
            continue
        c.execute('SELECT * FROM alpha_ids ORDER BY ABS( alpha - %s) ASC LIMIT 1' % str(a))
        id, value = c.fetchone()
        if(abs(value - a) < .001):
            mapping[id] = value
        else:
            c.execute('SELECT MAX(alpha_id) FROM alpha_ids')
            new_id, = c.fetchone()
            new_id += 1
            mapping[new_id] = a
            c.execute('INSERT INTO alpha_ids VALUES (%d,%s)' % (new_id, str(a)))
    conn.commit()
    conn.close()
    return mapping


def define_schemas():
    text = "TEXT"
    text_primary = "TEXT PRIMARY KEY"
    integer = "INTEGER"
    int_primary = "INTEGER PRIMARY KEY"
    real = "REAL"
    primary = "PRIMARY KEY"

    schemas = {}
    schemas["proximity_vectors"] = [("network_name", text), ("first_node", integer),
                                    ("second_node", integer), ("ranking", integer),
                                    ("alpha_id", integer), ("score", real), ("PRIMARY KEY", "(network_name, first_node, second_node, alpha_id)")]

    schemas["results"] = [("network_name", text), ("run_id", integer), ("query_size", integer),
                          ("alpha_id", integer), ("cache_size", integer), ("norm_type", text),
                          ("num_iterations", integer), ("walltime", integer)]

    schemas["alpha_ids"] = [("alpha_id", int_primary), ("alpha", real)]

    return schemas


def initialize_database(db_file, schemas):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    for table in schemas.keys():
        schema_columns = ", ".join("%s %s" % x for x in schemas[table])
        command = 'create table if not exists %s (%s)' % (table, schema_columns)
        print(command)
        c.execute(command)

    conn.commit()
    conn.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds cache of stored proximity vectors")
    parser.add_argument('--db_file', type=str)
    parser.add_argument('--matrix_file', type=str)
    parser.add_argument('--alphas', type=float, nargs='+')
    parser.add_argument('--num_threads', type=int)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--max_dim', type=int)
    parser.set_defaults(db_file="Cache/proximity_vectors.sqlite3", matrix_file="Data/Email-Enron.mat",
                        alphas=[.01, .1, .25, .5, .9], num_threads=5, top_k=200)
    args = parser.parse_args()

    matrix_file = args.matrix_file
    network_name = matrix_file[matrix_file.find("/") + 1:matrix_file.find(".")]
    kwargs = dict(num_threads=args.num_threads, top_k=args.top_k, max_dim=args.max_dim)

    build_cache(args.db_file, matrix_file, network_name, args.alphas, **kwargs)
