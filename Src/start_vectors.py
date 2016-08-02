
def unnormalized(vector_list):
    return sum(vector_list)


def total_sum(vector_list):
    vector = sum(vector_list)
    return vector / vector.sum()


def twice_normalized(vector_list):
    normalized_vectors = [v / v.sum() for v in vector_list]
    vector = sum(normalized_vectors)
    return vector / vector.sum()
