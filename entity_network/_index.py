import pandas as pd

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

    if df.index.has_duplicates:
        raise _exceptions.DuplicatedIndex('Argument df index must be unique.')
    if df2 is not None and df2.index.has_duplicates:
        raise _exceptions.DuplicatedIndex('Argument df2 index must be unique.')

    # develop unique integer based index for each df in case they need to be combined
    index_mask = {
        'df': pd.Series(df.index, index=range(0, len(df))),
        'df2': None
    }
    index_mask['df'].name = 'df_index'
    df = df.copy()
    df.index = index_mask['df'].index
    if df2 is not None:
        # start df2 index at end of df index
        seed = len(index_mask['df'])+1
        index_mask['df2'] = pd.Series(df2.index, index=range(seed, seed+len(df2)))
        index_mask['df2'].name = 'df2_index'
        df2 = df2.copy()
        df2.index = index_mask['df2'].index
        # stack df2 on each of df for a single df to be compared
        df = pd.concat([df, df2])

    return df, index_mask


def original(index_mask, df_id, df_feature):

    # add original index in id dataframe
    df_id = df_id.merge(index_mask['df'], left_on='index', right_index=True, how='left')
    df_id['df_index'] = df_id['df_index'].astype('Int64')
    if index_mask['df2'] is not None:
        df_id = df_id.merge(index_mask['df2'], left_on='index', right_index=True, how='left')
        df_id['df2_index'] = df_id['df2_index'].astype('Int64')

    # add original index in feature dataframe
    df_feature = df_feature.apply(pd.Series.explode)
    df_feature = df_feature.reset_index()
    df_feature = df_feature.merge(df_id, on='index')

    # remove artifical create index
    df_id = df_id.set_index('index')
    df_id.index.name = None
    df_feature = df_feature.set_index('index')
    df_feature.index.name = None

    return df_id, df_feature
