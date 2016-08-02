network_filepath="Data/test.mtx"
db_filepath="Cache/test.sqlite3"
cache_subdir="Cache/Test/"
num_threads=1
num_permutations=2
alphas=(.01 .1 .25)
cache_sizes=(10 50 200)
python Src/build_cache.py --network_filepath $network_filepath --db_filepath $db_filepath --cache_subdir $cache_subdir --num_threads $num_threads
python Src/generate_results.py --network_filepath $network_filepath --db_filepath $db_filepath --num_threads $num_threads --num_permutations $num_permutations --query_sizes 10 20 50