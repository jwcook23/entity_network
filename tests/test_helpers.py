import pandas as pd

from entity_network import _helpers

from scipy.sparse import bsr_array

# def test_large_series_graph():

edges = pd.Series([
    range(0,5000),
    (0,2)
])

# graph = _helpers.series_graph(edges)


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

graph = [
[0, 1, 1, 0, 0],
[0, 0, 1, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 1],
[0, 0, 0, 0, 0]
]

graph = csr_matrix(graph)

n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)