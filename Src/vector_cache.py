import ppr


class vector_cache:

    def __init__(self):
        self.cache = {}
        self.trimmed_cache = {}

    def get_vector(self, node_id, alpha):
        encoded_alpha = self.encode_alpha(alpha)
        return self.cache[(node_id, encoded_alpha)]

    def get_trimmed_vector(self, node_id, alpha, cache_size):
        encoded_alpha = self.encode_alpha(alpha)
        trimmed_key = (node_id, encoded_alpha, cache_size)
        standard_key = (node_id, encoded_alpha)

        if(trimmed_key in self.trimmed_cache.keys()):
            return self.trimmed_cache[trimmed_key]
        elif(standard_key in self.cache.keys()):
            trimmed_vector = ppr.trim_vector(self.cache[standard_key], cache_size)
            self.trimmed_cache[trimmed_key] = trimmed_vector
            return trimmed_vector

    def insert_vector(self, node_id, alpha, vector):
        encoded_alpha = self.encode_alpha(alpha)
        self.cache[(node_id, encoded_alpha)] = vector

    #Encodes alpha as a string
    #This is so that alpha (a floating point number) can be used as a dictionary key
    #Simply takes the first three points after the decimal point
    def encode_alpha(self, alpha):
        return "%.3f" % alpha

    def get_vector_list(self, query_set, alpha, cache_size=None):
        if(cache_size):
            return [self.get_trimmed_vector(node_id, alpha, cache_size) for node_id in query_set]
        else:
            return [self.get_vector(node_id, alpha) for node_id in query_set]

    def __str__(self):
        return "\n".join("(%d, %s): %s %d" % (key[0], key[1], str(type(self.cache[key])), i) for i, key in enumerate(self.cache.keys()))
