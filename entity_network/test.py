from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components

values = [[0,1],[0,2],[1,2],[3,4]]
# rows, cols = zip(*values)
# vals = [1]*len(rows)
# graph = csr_matrix((vals, (rows, cols)), shape=(5,5))
graph = lil_matrix((5,5), dtype=int)
for row_index, col_indices in enumerate(values):
    graph[row_index, col_indices] = 1
n_components, labels = connected_components(graph, directed=False, return_labels=True)


# https://stackoverflow.com/questions/18453163/list-of-numpy-vectors-to-sparse-array
values = [[0,1], [0,2,3], [4,5], [6,7], [8,9], [7,10]]

# graph = [
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
#     [0,0,0,0,0,0],
# ]


graph = lil_matrix((11,11), dtype=int)
for idx in values:
    graph[min(idx), idx] = 1
# graph = graph.tocsr()