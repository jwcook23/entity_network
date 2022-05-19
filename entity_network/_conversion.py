from itertools import combinations
import networkx as nx


def from_pandas_series(edges):
    '''Convert Pandas series of lists to graph.'''

    edges = edges.apply(lambda x: list(combinations(x,2)))

    graph = nx.Graph()
    for e in edges.values:
        graph.add_edges_from(e)
    
    return graph

