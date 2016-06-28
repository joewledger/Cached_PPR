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


def read_csr_matrix(filename):
    return spio.loadmat(filename)["normalizedNetwork"]
