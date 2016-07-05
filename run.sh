data_file="Data/Email-Enron.mat"
cache_file="Cache/cached_ppr.sqlite3"
alphas=(.01 .1 .25)
cache_sizes=(10 50 200)
python Src/build_cache.py
python Src/generate_results.py
python Src/plotting.py