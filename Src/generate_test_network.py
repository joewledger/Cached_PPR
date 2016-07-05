import random
from scipy.sparse import dok_matrix
import scipy.io as spio
from sklearn.preprocessing import normalize
import sys

size = 100
savefile = sys.argv[1]

connections = set()
all_nodes = {x for x in range(0, size)}
connected = set(random.sample(all_nodes, 1))
unconnected = all_nodes - connected

while(len(unconnected) > 0):
    first_choice = random.sample(unconnected, 1)[0]
    second_choice = random.sample(all_nodes, 1)[0]
    connections.add((first_choice, second_choice))
    connections.add((second_choice, first_choice))
    if(second_choice in connected):
        connected.add(first_choice)
        unconnected.remove(first_choice)

matrix = dok_matrix((size, size))
for entry in connections:
    matrix[entry[0], entry[1]] = 1.0

matrix = normalize(matrix, norm='l1', axis=0)
matrix = matrix.tocsr()
spio.mmwrite(savefile, matrix)
