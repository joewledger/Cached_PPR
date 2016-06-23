import build_cache
import ppr
import sqlite3


def test_save_proximity_vector():

    db_file = "Cache/test7.sqlite3"
    matrix_file = "Data/Email-Enron.mat"
    weight_matrix = ppr.read_csr_matrix(matrix_file)
    network_name = "Enron"
    node_id = 5
    alpha_id = 0
    alpha_value = .5
    top_k = 100

    build_cache.initialize_database(db_file, build_cache.define_schemas())
    build_cache.save_proximity_vector(db_file, weight_matrix, network_name, node_id, alpha_id, alpha_value, top_k)

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT * FROM proximity_vectors')
    print(c.fetchone())
    conn.close()


def test_update_alpha_ids():

    db_file = "Cache/test10.sqlite3"
    alphas = [.01, .1, .2, .5]
    #try:
    #    initialize_alpha_id_table(db_file, alphas)
    #except:
    #    print("Table already exists")

    alphas.extend([.5000001, .75])
    mapping = build_cache.update_alpha_ids(db_file, alphas)
    print(mapping)


def initialize_alpha_id_table(db_file, alphas):

    build_cache.initialize_database(db_file, build_cache.define_schemas())
    values = [(i, x) for i, x in enumerate(alphas)]

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.executemany("INSERT INTO alpha_ids VALUES (?,?)", values)
    conn.commit()
    conn.close()
