import random


def get_query_sets(num_sets, set_size, population):
    return [random.sample(population, set_size) for _ in range(num_sets)]


def unnormalized(vector_list):
    return sum(vector_list)


def total_sum(vector_list):
    vector = sum(vector_list)
    return vector / vector.sum()


def twice_normalized(vector_list):
    normalized_vectors = [v / v.sum() for v in vector_list]
    vector = sum(normalized_vectors)
    return vector / vector.sum()
