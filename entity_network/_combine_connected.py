import pandas as pd

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

def combine_features(relationships):
    '''Combine records with matching feature ids.'''

    network_map = pd.DataFrame()
    for category, related in relationships.items():
        if category=='name':
            # use all features besides name for forming network
            continue
        id_category = f'{category}_id'
        related = related[[id_category]].copy()
        network_map = network_map.merge(related, how='outer', left_index=True, right_index=True)
        network_map[id_category] = network_map[id_category].astype('Int64')
    network_map = network_map[network_map.notna().any(axis='columns')]
    network_map = network_map.reset_index()

    return network_map

def overall_id(network_map):
    
    # assume an initial temp id to find connected components of
    id_cols = network_map.columns[network_map.columns.str.endswith('_id')]
    network_map['temp_id'] = range(0, len(network_map))

    # form list of lists sparse matrix incrementally of connected nodes
    # TODO: form this sparse matrix during initial comparison to increase performance
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
    # TODO: remove network_id and keep network_map only?
    network_id = network_map[['node','network_id']]
    network_id = network_id.drop_duplicates(subset='node')

    return network_id, network_map
