
class vector_cache:

    def __init__(self):
        self.cache = {}

    def get_vector(self, node_id, alpha):
        encoded_alpha = self.encode_alpha(alpha)
        return self.cache[(node_id, encoded_alpha)]

    def insert_vector(self, node_id, alpha, vector):
        encoded_alpha = self.encode_alpha(alpha)
        self.cache[(node_id, encoded_alpha)] = vector

    #Encodes alpha as a string
    #This is so that alpha (a floating point number) can be used as a dictionary key
    #Simply takes the first three points after the decimal point
    def encode_alpha(self, alpha):
        return "%.3f" % alpha

    def __str__(self):
        return str(len(self.cache.values()))
