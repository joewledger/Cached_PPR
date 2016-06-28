import scipy.io as spio
import sqlite3


#Returns {first_node -> {second_node -> score}}
def load_cached_vectors(db_file, query_nodes, alpha, cache_size):

    cached_vectors = {}

    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    c.execute('SELECT alpha_id FROM alpha_ids ORDER BY ABS( alpha - %s) ASC LIMIT 1' % str(alpha))
    alpha_id, = c.fetchone()

    for q in query_nodes:
        cached_vectors[q] = {}
        c.execute('SELECT second_node, score FROM proximity_vectors WHERE first_node = %d AND alpha_id = %d ORDER BY score DESC LIMIT %d' % (q, alpha_id, cache_size))
        for second_node, score in c.fetchall():
            cached_vectors[q][second_node] = score

    conn.commit()
    conn.close()

    return cached_vectors


def get_closest_alpha_id(db_file, alpha):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT alpha_id FROM alpha_ids ORDER BY ABS( alpha - %s) ASC LIMIT 1' % str(alpha))
    alpha_id, = c.fetchone()
    conn.commit()
    conn.close()
    return alpha_id


def get_network_name(matrix_file):
    return matrix_file[matrix_file.find("/") + 1:matrix_file.find(".")]


def read_csr_matrix(filename):
    return spio.loadmat(filename)["normalizedNetwork"]


def save_results(db_file, results_generator):

    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    for result in results_generator:
        result[0] = get_network_name(result[0])
        result[3] = get_closest_alpha_id(db_file, result[3])
        result[5] = (result[5] if result[5] else "Unnormalized")
        c.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?)', result)
        conn.commit()
    conn.close()


def get_num_iterations(db_file, alpha=None, query_size=None, cache_size=None, norm_method=None):

    query = 'SELECT num_iterations FROM results'
    clauses = []
    if(alpha):
        alpha_id = get_closest_alpha_id(db_file, alpha)
        clauses.append('alpha_id = %d' % alpha_id)
    if(query_size):
        clauses.append('query_size = %d' % query_size)
    if(cache_size):
        clauses.append('cache_size = %d' % cache_size)
    if(norm_method):
        clauses.append('norm_type = "%s"' % norm_method)

    if(len(clauses) > 0):
        query += ' WHERE ' + ' AND '.join(x for x in clauses) + ';'

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(query)
    results = [x[0] for x in c.fetchall()]
    conn.close()

    return results
