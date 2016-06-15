import build_cache
import ppr

def test_build_cache():
	
	cache_name = "Email-Enron"
	alpha = .5
	path = "Cache/%s/%s/" % (cache_name, str(alpha))
	weight_matrix = ppr.read_csr_matrix("Data/%s.mat" % cache_name)
	cache_nodes = range(0,5)

	build_cache.build_cache(path,weight_matrix,alpha,cache_nodes=cache_nodes)


def test_cache_equivalence():
	return None