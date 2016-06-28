import random


def generate_query_sets(query_size, query_range, num_permutations):
    return [random.sample(range(0, query_range), query_size) for n in range(0, num_permutations)]

print(generate_query_sets(10, 20, 5))
