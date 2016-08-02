
def unnormalized_start_vector(vector_list):
    return sum(vector_list)


def total_sum_vector(vector_list):
    vector = sum(vector_list)
    return vector / vector.sum()


def twice_normalized_vector(vector_list):
    normalized_vectors = [v / v.sum() for v in vector_list]
    vector = sum(normalized_vectors)
    return vector / vector.sum()
