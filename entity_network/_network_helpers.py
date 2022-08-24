from unicodedata import category
import pandas as pd

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from entity_network import _index
from entity_network import _find_difference

def combine_features(relationships):
    '''Combine records with matching feature ids.'''

    network_map = pd.DataFrame(columns=['node'])
    for category, related in relationships.items():
        if category=='name':
            # use all features besides name for forming network
            continue
        id_category = f'{category}_id'
        related = related[[id_category]].copy()
        related = related.reset_index()
        network_map = network_map.merge(related, how='outer', on='node')
        network_map[id_category] = network_map[id_category].astype('Int64')
    
    network_map = network_map[network_map.drop(columns='node').notna().any(axis='columns')]

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

    # summerize network by index
    network_summary = network_id.groupby('network_id')
    list_notna = lambda x: list(x.dropna())
    network_summary = network_summary.agg({'df_index': list_notna, 'df2_index': list_notna})
    network_summary['feature'] = ''

    # find matched categories and differences between values
    column_feature = []
    for category, feature in network_feature.items():
        # determine column name and add processed values
        feature = feature[['column']]
        df = feature.merge(processed[category]['df'], on=['node','column'], how='inner')
        df['source'] = 'df'
        df2 = feature.merge(processed[category]['df2'], on=['node','column'], how='inner')
        df2['source'] = 'df2'
        feature = pd.concat([df,df2])
        feature = feature.merge(network_id[['network_id']], on='node', how='inner')
        # group values by id and calculate term differences
        feature = feature.set_index('network_id')
        feature = _find_difference.main(feature, category)
        # set a matching feature column
        column = f'{category}_feature'
        column_feature+=[column]
        feature[column] = category

        # add values into network summary
        network_summary = network_summary.merge(feature, on='network_id', how='left')
        empty = feature.columns.drop(column)
        network_summary[empty] = network_summary[empty].where(pd.notnull(network_summary[empty]), None)
        network_summary[column] = network_summary[column].fillna('')

    # form a single comma seperated feature column
    network_summary[column_feature] = network_summary[column_feature].fillna('')
    network_summary['feature'] = network_summary[column_feature].apply(lambda x: ','.join(x), axis='columns')
    network_summary['feature'] = network_summary['feature'].str.replace(',{2,}', ',', regex=True)
    network_summary['feature'] = network_summary['feature'].str.replace('^,|,$', '', regex=True)
    network_summary = network_summary.drop(columns=column_feature)

    return network_summary
