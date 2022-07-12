import pandas as pd

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

def combine_features(relationships, indices):
    '''Combine records with matching feature ids.'''

    network_map = pd.DataFrame(index=indices)
    for category, related in relationships.items():
        id_category = f'{category}_id'
        related = related[[id_category]].copy()
        network_map = network_map.merge(related, how='left', left_index=True, right_index=True)
        network_map[id_category] = network_map[id_category].astype('Int64')
    network_map = network_map[network_map.notna().any(axis='columns')]
    network_map.index.name = 'node'
    network_map = network_map.reset_index()

    return network_map

def overall_id(network_map):
    
    # assume an initial temp id to find connected components of
    id_cols = network_map.columns[network_map.columns.str.endswith('_id')]
    network_map['temp_id'] = range(0, len(network_map))

    # form list of lists sparse matrix incrementally of connected nodes
    n = len(network_map)
    graph = lil_matrix((n, n), dtype=int)
    for col in id_cols:
        indices = network_map.groupby(col)
        indices = indices.agg({'temp_id': list})
        indices = indices['temp_id']
        for idx in indices:
            graph[idx[0], idx] = 1
    
    # convert to compressed sparse row matrix for better computation
    graph = graph.tocsr()

    # determine connected temp_ids to assign an overall network_id
    _, labels = connected_components(graph, directed=False, return_labels=True)
    network_map['network_id'] = labels
    network_map = network_map.drop(columns='temp_id')

    # determine overall network id
    network_id = network_map[['node','network_id']]
    network_id = network_id.drop_duplicates(subset='node')

    return network_id, network_map

# def overall_id(network_map):
#     '''Assign a group_id to rows sharing a common column value.'''
    
#     # extract values for numpy comparison
#     columns = network_map.columns[network_map.columns.str.endswith('_id')]
#     values = network_map[columns].to_numpy(na_value=np.nan)
#     values = np.vstack(values[:, :]).astype(np.float)

#     # compare the difference of all elements
#     matches = (values[:,None]-values)==0
#     # find matches in other rows (each row corresponds to the row in A, each column is the comparison to other rows in A)
#     matches = matches.any(axis=2)

#     # find indices of match groups
#     groups = np.argwhere(matches)
#     # remove self matches in groups
#     groups = groups[groups[:,0]!=groups[:,1]]
#     # remove duplicates in groups
#     groups.sort(axis=1)
#     groups = np.unique(groups, axis=0)

#     # assign the same group id to row groups
#     group_id = np.arange(values.shape[0])
#     for row_pair in groups:
#         row0 = row_pair[0]
#         row1 = row_pair[1]
#         assign = np.flatnonzero(group_id==group_id[row1])
#         group_id[assign] = group_id[row0]

#     # assign id to input df
#     network_map['network_id'] = group_id
#     network_map['network_id'] = network_map['network_id'].astype('int64')

#     # determine overall network id
#     network_id = network_map[['node','network_id']]
#     network_id = network_id.drop_duplicates(subset='node')

#     return network_id, network_map


# def _common_features(relationships):

#     network_map = []
#     for category,related in relationships.items():

#         category_id = f'{category}_id'
#         feature = related.reset_index()

#         # aggreate values to form network
#         feature = feature.groupby(category_id)
#         feature = feature.agg({'node': tuple, 'column': tuple})

#         # remove records that only match the same record
#         feature = feature.loc[
#             feature['node'].apply(lambda x: len(set(x))>1)
#         ]

#         # append details for category
#         network_map.append(feature)
#     # dataframe of all matching features
#     network_map = pd.concat(network_map, ignore_index=True)
#     network_map.index.name = 'feature_id'

#     return network_map

# def _series_graph(network_map):
#     '''Convert Pandas series of lists to graph.'''

#     edges = network_map['node'].apply(lambda x: list(combinations(x,2)))

#     edges = edges.explode()
#     edges = pd.DataFrame(edges.tolist(), columns=['source','target'])
#     graph = nx.from_pandas_edgelist(edges)

#     return graph

# def _assign_id(graph):

#     df_id = pd.DataFrame({'node': list(nx.connected_components(graph))})
#     df_id['network_id'] = range(0, len(df_id))
#     df_id = df_id.explode('node')

#     return df_id