from itertools import combinations

import pandas as pd
import networkx as nx

def assign_group(relationships, indices):

    # df_feature = _common_features(relationships)
    # graph = _series_graph(df_feature)
    # df_id = _assign_id(graph)

    df_id = pd.DataFrame(index=indices)
    for category, related in relationships.items():
        category_id = f'{category}_id'
        df_id = df_id.merge(related[category_id], how='left', left_index=True, right_index=True)
        df_id[category_id] = df_id[category_id].astype('Int64')
    df_id = df_id[df_id.notna().any(axis='columns')]

    # https://stackoverflow.com/questions/61763966/python-numpy-grouping-array-rows-by-a-common-element
    import numpy as np
    a = df_id.values
    a = np.array([
        [1,2,3],
        [0,4,2],
        [4,2,5],
        [6,1,1],
        [1,3,5],
        [3,0,1],
        [0,4,2],
        [1, np.nan, np.nan],
        [7, np.nan, np.nan]
    ])
    pairs = np.argwhere(((a[:,None]-a)==0).any(axis=2))
    b = np.arange(a.shape[0])
    for pair in pairs:
        b[np.flatnonzero(b==b[pair[1]])] = b[pair[0]]
    b = b - b.min()
    df_id['network_id'] = b

    df_feature = None

    return df_id, df_feature


def _common_features(relationships):

    df_feature = []
    for category,related in relationships.items():

        category_id = f'{category}_id'
        feature = related.reset_index()

        # aggreate values to form network
        feature = feature.groupby(category_id)
        feature = feature.agg({'index': tuple, 'column': tuple})

        # remove records that only match the same record
        feature = feature.loc[
            feature['index'].apply(lambda x: len(set(x))>1)
        ]

        # append details for category
        df_feature.append(feature)
    # dataframe of all matching features
    df_feature = pd.concat(df_feature, ignore_index=True)
    df_feature.index.name = 'feature_id'

    return df_feature

def _series_graph(df_feature):
    '''Convert Pandas series of lists to graph.'''

    edges = df_feature['index'].apply(lambda x: list(combinations(x,2)))

    edges = edges.explode()
    edges = pd.DataFrame(edges.tolist(), columns=['source','target'])
    graph = nx.from_pandas_edgelist(edges)

    return graph

def _assign_id(graph):

    df_id = pd.DataFrame({'index': list(nx.connected_components(graph))})
    df_id['network_id'] = range(0, len(df_id))
    df_id = df_id.explode('index')

    return df_id