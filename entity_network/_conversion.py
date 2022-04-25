from itertools import combinations
import networkx as nx

def from_pandas_df_id(data, id):
    '''Convert Pandas dataframe containing a column with lists and an id column to a graph.'''

    edges = data.groupby(id).agg({'index': 'unique'})
    edges = edges[edges['index'].str.len()>1]
    graph = from_pandas_series(edges['index'])

    return graph


def from_pandas_series(edges):
    '''Convert Pandas series of lists to graph.'''

    edges = edges.to_list()
    edges = [list(combinations(x,2)) for x in edges]
    graph = nx.Graph()
    for e in edges:
        graph.add_edges_from(e)
    
    return graph

