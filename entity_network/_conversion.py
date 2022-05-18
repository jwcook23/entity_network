from itertools import combinations
import networkx as nx


def from_pandas_df(df):
    '''Convert Pandas series of lists to graph.'''

    df['edges'] = df['index'].apply(lambda x: list(combinations(x,2)))

    graph = nx.Graph()
    for _,edge in df.iterrows():
        graph.add_edges_from(edge['edges'])
    
    return graph

