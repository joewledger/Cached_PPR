CREATE TABLE IF NOT EXISTS proximity_vectors (network_filepath TEXT, first_node INTEGER, alpha_id INTEGER, proximity_filepath TEXT, vector_length INTEGER, PRIMARY KEY (network_filepath, first_node, alpha_id));
CREATE TABLE IF NOT EXISTS alpha_ids (alpha_id INTEGER PRIMARY KEY, alpha REAL); 
CREATE TABLE IF NOT EXISTS query_sets (network_filepath TEXT, query_id INTEGER, query_set BLOB, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (network_filepath, query_id));
CREATE TABLE IF NOT EXISTS results (network_filepath TEXT, query_id INTEGER, alpha_id INTEGER, cache_size INTEGER, method_id INTEGER, final_vector BLOB, num_iterations INTEGER, error_terms BLOB, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (network_filepath, query_id, alpha_id, method_id));
CREATE TABLE IF NOT EXISTS method_ids (method_id INTEGER PRIMARY KEY, method TEXT);
INSERT INTO method_ids (method_id, method) VALUES (0, "Standard"), (1, "Unnormalized"), (2, "Total_Sum"), (3, "Twice_Normalized");