from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx

# def assign_group(relationships, indices):

#     # network_map = _common_features(relationships)
#     # graph = _series_graph(network_map)
#     # df_id = _assign_id(graph)

#     network_map = _combine_features(relationships, indices)
#     network_map = _row_connected(network_map)
#     df_id = _unique_id(network_map)

#     return df_id, network_map

def combine_features(relationships, indices):
    '''Combine records with matching feature ids.'''

    network_map = pd.DataFrame(index=indices)
    for category, related in relationships.items():
        id_category = f'{category}_id'
        related = related[[id_category]].copy()
        network_map = network_map.merge(related, how='left', left_index=True, right_index=True)
        network_map[id_category] = network_map[id_category].astype('Int64')
    network_map = network_map[network_map.notna().any(axis='columns')]
    network_map.index.name = 'index'
    network_map = network_map.reset_index()

    return network_map

def overall_id(network_map):
    '''Assign a group_id to rows sharing a common column value.'''
    
    # extract values for numpy comparison
    columns = network_map.columns[network_map.columns.str.endswith('_id')]
    values = network_map[columns].to_numpy(na_value=np.nan)
    values = np.vstack(values[:, :]).astype(np.float)

    # compare the difference of all elements
    matches = (values[:,None]-values)==0
    # find matches in other rows (each row corresponds to the row in A, each column is the comparison to other rows in A)
    matches = matches.any(axis=2)

    # find indices of match groups
    groups = np.argwhere(matches)
    # remove self matches in groups
    groups = groups[groups[:,0]!=groups[:,1]]
    # remove duplicates in groups
    groups.sort(axis=1)
    groups = np.unique(groups, axis=0)

    # assign the same group id to row groups
    group_id = np.arange(values.shape[0])
    for row_pair in groups:
        row0 = row_pair[0]
        row1 = row_pair[1]
        assign = np.flatnonzero(group_id==group_id[row1])
        group_id[assign] = group_id[row0]

    # assign id to input df
    network_map['network_id'] = group_id
    network_map['network_id'] = network_map['network_id'].astype('int64')

    # determine overall network id
    network_id = network_map[['index','network_id']]
    network_id = network_id.drop_duplicates(subset='index')

    return network_id, network_map


def flatten_related():

    pass


# def _common_features(relationships):

#     network_map = []
#     for category,related in relationships.items():

#         category_id = f'{category}_id'
#         feature = related.reset_index()

#         # aggreate values to form network
#         feature = feature.groupby(category_id)
#         feature = feature.agg({'index': tuple, 'column': tuple})

#         # remove records that only match the same record
#         feature = feature.loc[
#             feature['index'].apply(lambda x: len(set(x))>1)
#         ]

#         # append details for category
#         network_map.append(feature)
#     # dataframe of all matching features
#     network_map = pd.concat(network_map, ignore_index=True)
#     network_map.index.name = 'feature_id'

#     return network_map

# def _series_graph(network_map):
#     '''Convert Pandas series of lists to graph.'''

#     edges = network_map['index'].apply(lambda x: list(combinations(x,2)))

#     edges = edges.explode()
#     edges = pd.DataFrame(edges.tolist(), columns=['source','target'])
#     graph = nx.from_pandas_edgelist(edges)

#     return graph

# def _assign_id(graph):

#     df_id = pd.DataFrame({'index': list(nx.connected_components(graph))})
#     df_id['network_id'] = range(0, len(df_id))
#     df_id = df_id.explode('index')

#     return df_id