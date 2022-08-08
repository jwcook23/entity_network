import re
import pandas as pd
from pandas.api.types import is_numeric_dtype

from entity_network import clean_text, _exceptions

default_text_cleaner = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}
default_text_comparer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def unique(df, df2):

    # make soft copy to preserve originals
    df = df.copy()
    if df2 is not None:
        df2 = df2.copy()

    # enforce unique values for tracking values
    if df.index.has_duplicates:
        raise _exceptions.DuplicatedIndex('Argument df index must be unique.')
    if df2 is not None and df2.index.has_duplicates:
        raise _exceptions.DuplicatedIndex('Argument df2 index must be unique.')

    # form a single dataframe container
    dfs = {'df': df, 'df2': df2}

    # develop unique integer based index
    index_mask = {
        'df': pd.Series(df.index, index=range(0, len(df))),
        'df2': None
    }
    index_mask['df'].name = 'df_index'
    dfs['df'].index = index_mask['df'].index
    dfs['df'].index.name = 'node'
    
    # set index of df2 starting at end of df 
    if df2 is not None:
        seed = len(index_mask['df'])
        index_mask['df2'] = pd.Series(df2.index, index=range(seed, seed+len(df2)))
        index_mask['df2'].name = 'df2_index'
        dfs['df2'].index = index_mask['df2'].index
        dfs['df2'].index.name = 'node'

    # df = df.copy()
    # df.index = index_mask['df'].index
    # if df2 is not None:
    #     # start df2 index at end of df index
    #     seed = len(index_mask['df'])
    #     index_mask['df2'] = pd.Series(df2.index, index=range(seed, seed+len(df2)))
    #     index_mask['df2'].name = 'df2_index'
    #     df2 = df2.copy()
    #     df2.index = index_mask['df2'].index
    #     # stack df2 on each of df for a single df to be compared
    #     df = pd.concat([df, df2])

    return dfs, index_mask

def recast(reindexed, mask):
    
    if is_numeric_dtype(mask):
        dtype = str(mask.dtype).capitalize()
    else:
        dtype = str(mask.dtype)
    reindexed[mask.name] = reindexed[mask.name].astype(dtype)

    return reindexed

def original(reindexed, index_mask):

    # if index_name=='node_similar':
    #     mask.name = f'{mask.name}_similar'

    # include the node index from the first df
    mask = index_mask['df'].copy()
    reindexed = reindexed.merge(mask, left_on='node', right_index=True, how='left')
    reindexed = recast(reindexed, mask)
    if 'node_similar' in reindexed and reindexed['node_similar'].isin(mask.index).any():
        reindexed = reindexed.merge(mask, left_on='node_similar', right_index=True, how='left', suffixes=('','_similar'))
        reindexed = recast(reindexed, mask)

    # add index from the second dataframe
    if index_mask['df2'] is not None:
        mask = index_mask['df2'].copy()
        if 'node_similar' in reindexed:
            reindexed = reindexed.merge(mask, left_on='node_similar', right_index=True, how='left')
            reindexed = recast(reindexed, mask)
        else:
            reindexed = reindexed.merge(mask, left_on='node', right_index=True, how='left')
            reindexed = recast(reindexed, mask)

    return reindexed


def network(df_id, df_feature, index_mask):

    # add original index in id dataframe
    df_id = original(df_id, index_mask)

    # add original index in feature dataframe
    df_feature = df_feature.merge(df_id.drop(columns=['network_id']), on='node')

    # remove artifical create index
    df_id = df_id.set_index('node')
    df_feature = df_feature.set_index('node')

    return df_id, df_feature
