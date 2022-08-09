import re
import pandas as pd
from pandas.api.types import is_numeric_dtype

from entity_network import _exceptions


def assign_node(df, df2):
    '''Assign each record a unique node value. This allows for comparison using any input index data type.'''

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

    return dfs, index_mask


def _preserve_type(reindexed, mask):
    '''Translate index back into the original data type and allow for nullable numeric types.'''

    if is_numeric_dtype(mask):
        dtype = str(mask.dtype).capitalize()
    else:
        dtype = str(mask.dtype)
    reindexed[mask.name] = reindexed[mask.name].astype(dtype)

    return reindexed

def assign_index(reindexed, index_mask):

    # include the node index from the first df
    mask = index_mask['df'].copy()
    reindexed = reindexed.merge(mask, left_on='node', right_index=True, how='left')
    reindexed = _preserve_type(reindexed, mask)
    if 'node_similar' in reindexed and reindexed['node_similar'].isin(mask.index).any():
        reindexed = reindexed.merge(mask, left_on='node_similar', right_index=True, how='left', suffixes=('','_similar'))
        reindexed = _preserve_type(reindexed, mask)

    # add index from the second dataframe
    if index_mask['df2'] is not None:
        mask = index_mask['df2'].copy()
        if 'node_similar' in reindexed:
            reindexed = reindexed.merge(mask, left_on='node_similar', right_index=True, how='left')
            reindexed = _preserve_type(reindexed, mask)
        else:
            reindexed = reindexed.merge(mask, left_on='node', right_index=True, how='left')
            reindexed = _preserve_type(reindexed, mask)

    return reindexed
