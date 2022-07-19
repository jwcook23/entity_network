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


def original(reindexed, index_mask, index_name = 'node'):

    mask = index_mask['df'].copy()
    if index_name=='node_similar':
        mask.name = f'{mask.name}_similar'
    reindexed = reindexed.merge(mask, left_on=index_name, right_index=True, how='left')
    if is_numeric_dtype(index_mask['df']):
        dtype = str(index_mask['df'].dtype).capitalize()
    else:
        dtype = str(index_mask['df'].dtype)
    reindexed[mask.name] = reindexed[mask.name].astype(dtype)
    if index_mask['df2'] is not None:
        mask = index_mask['df2'].copy()
        if index_name=='node_similar':
            mask.name = f'{mask.name}_similar'        
        reindexed = reindexed.merge(mask, left_on=index_name, right_index=True, how='left')
        if is_numeric_dtype(index_mask['df2']):
            dtype = str(index_mask['df2'].dtype).capitalize()
        else:
            dtype = str(index_mask['df2'].dtype)
        reindexed[mask.name] = reindexed[mask.name].astype(dtype)

    return reindexed


def network(df_id, df_feature, index_mask):

    # add original index in id dataframe
    # df_id = df_id.merge(index_mask['df'], left_on='node', right_index=True, how='left')
    # df_id['df_index'] = df_id['df_index'].astype('Int64')
    # if index_mask['df2'] is not None:
    #     df_id = df_id.merge(index_mask['df2'], left_on='node', right_index=True, how='left')
    #     df_id['df2_index'] = df_id['df2_index'].astype('Int64')
    df_id = original(df_id, index_mask)

    # add original index in feature dataframe
    # df_feature = df_feature.apply(pd.Series.explode)
    # df_feature = df_feature.reset_index()
    df_feature = df_feature.merge(df_id.drop(columns=['network_id']), on='node')

    # remove artifical create index
    df_id = df_id.set_index('node')
    df_feature = df_feature.set_index('node')

    return df_id, df_feature
