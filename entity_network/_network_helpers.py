import pandas as pd

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from entity_network import _index

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

def assign_id(network_map):
    
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

def translate_index(network_id, network_map, index_mask):

    network_id = _index.assign_index(network_id, index_mask)
    network_map = network_map.merge(network_id.drop(columns=['network_id']), on='node')
    network_id = network_id.set_index('node')
    network_map = network_map.set_index('node')

    return network_id, network_map

def resolve_entity(network_map, network_feature, df):
    # assume similar names in the same network are the same entity
    # TODO: require a single matching feature instead of the entire network?
    entity_id = network_map[['network_id']].merge(
        network_feature['name'][['name_id']], 
        left_index=True, right_index=True, how='left'
    )
    entity_id = entity_id.groupby(['network_id','name_id']).ngroup()

    # assume entity_id without a name_id
    assume = entity_id==-1
    seed = entity_id.max()+1
    if pd.isna(seed):
        seed = 0
    entity_id[assume] = range(seed, seed+sum(assume))

    # assign resolved entity to network map
    network_map['entity_id'] = entity_id

    # determine unique features belonging to each entity
    entity_map = pd.DataFrame(columns=['network_id','entity_id','category','column','value'])
    for category, feature in network_feature.items():
        category_id = f'{category}_id'
        # determine matching column
        details = network_map[['network_id','entity_id']]
        details = details.merge(feature[['column', category_id]], on='node', how='left')
        # add value from matching column
        # TODO: handle possibly two dataframes
        columns = details['column'].dropna().unique()
        value = df[columns].stack()
        value.name = 'value'
        details = details.merge(value, left_on=['node','column'], right_index=True)
        # remove duplicated info
        details = details.drop_duplicates()
        # add source category
        details['category'] = category
        # combine details from each features
        entity_map = pd.concat([entity_map, details])
        entity_map[category_id] = entity_map[category_id].astype('Int64')

    return entity_map, network_map


def summerize_entity(network_map, compared_columns, df):

    # rename / create columns for pandas aggregation
    network_summary = network_map.copy()
    columns = network_summary.columns.drop('network_id')
    columns = columns[columns.str.endswith('_id')]
    renamed = {c:c.replace('_id',' Count').capitalize() for c in columns}
    network_summary = network_summary.rename(columns=renamed)
    network_summary['Entity common'] = network_summary['Entity count']

    # aggreate number of unique values and most common value
    agg = {k:'nunique' for k in renamed.values()}
    agg['Entity common'] = pd.Series.mode
    network_summary = network_summary.groupby('network_id')
    network_summary = network_summary.agg(agg)

    # add the common name to the network summary

    # sort by largest network of entities
    network_summary = network_summary.sort_values('entity_count', ascending=False)

    return network_summary


def summerize_connections(network_id, network_feature, processed):
    
    # TODO: implement single df summary
    if 'df2_index' not in network_id:
        network_summary = None
        return network_summary

    # use the first dataframe as the primary dataframe to summerize
    network_summary = network_id[['network_id','df_index']].dropna()
    network_summary = network_summary.groupby('network_id').agg({'df_index': list})

    # add index from the second dataframe that are in the same network in the first dataframe
    other = network_id[['network_id','df2_index']].dropna()
    other = other.groupby('network_id').agg({'df2_index': list})
    network_summary = network_summary.merge(other, left_index=True, right_index=True, how='inner')

    # expand indices to add details
    network_summary = network_summary.explode('df2_index')
    network_summary = network_summary.explode('df_index')

    # determine the category(s) matched and add the processed values
    for category, data in network_feature.items():
        for frame in ['df','df2']:
            # select values from the current dataframe
            source = data[[f'{frame}_index']].dropna()
            if len(source)>0:
                # add processed values
                source = source.merge(processed[category][frame], on='node')
                source.columns = [f'{frame}_index', f'{frame}_value']
                # add category label
                source[f'{frame}_feature'] = category
                # aggreate by frame index
                source = source.groupby(f'{frame}_index')
                source = source.agg({f'{frame}_feature': list, f'{frame}_value': list})
                network_summary = network_summary.merge(source, on=f'{frame}_index', how='left')

    # add a combined list of matching category(s) concatenated with a comma
    network_summary['df_feature'] = network_summary['df_feature'].apply(lambda x: ','.join(x))
    network_summary['df2_feature'] = network_summary['df2_feature'].apply(lambda x: ','.join(x))

    return network_summary