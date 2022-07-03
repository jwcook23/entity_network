from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx

def assign_group(relationships, indices):

    # df_feature = _common_features(relationships)
    # graph = _series_graph(df_feature)
    # df_id = _assign_id(graph)

    df_feature = _combine_features(relationships, indices)
    df_feature = _row_connected(df_feature)
    df_id = _unique_id(df_feature)

    return df_id, df_feature

def _combine_features(relationships, indices):
    '''Combine records with matching feature ids.'''

    df_feature = pd.DataFrame(index=indices)
    for category, related in relationships.items():
        category_id = f'{category}_id'
        feature_name = f'{category}_feature'
        related = related[['id', 'column']].copy()
        related = related.rename(columns={'id': category_id, 'column': feature_name})
        df_feature = df_feature.merge(related, how='left', left_index=True, right_index=True)
        df_feature[category_id] = df_feature[category_id].astype('Int64')
    df_feature = df_feature[df_feature.notna().any(axis='columns')]
    df_feature.index.name = 'index'
    df_feature = df_feature.reset_index()

    return df_feature

def _row_connected(df):
    '''Assign a group_id to rows sharing a common column value.'''
    
    # extract values for numpy comparison
    columns = df.columns[df.columns.str.endswith('_id')]
    values = df[columns].to_numpy(na_value=np.nan)
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
    df['network_id'] = group_id
    df['network_id'] = df['network_id'].astype('int64')

    return df

def _unique_id(df_feature):
    '''Exact unique network_id for each index.'''

    df_id = df_feature[['index','network_id']]
    df_id = df_id.drop_duplicates(subset='index')
    
    return df_id


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