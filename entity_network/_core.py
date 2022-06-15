import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
import networkx as nx

from entity_network import clean_text, _conversion, _exceptions

default_preprocessor = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}
default_analyzer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def prepare_index(df, df2):

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


def original_id_index(index_mask, df_id, df_feature):

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


def prepare_values(df, category, columns, preprocessor):

    # check allowed category argument
    if not category in default_preprocessor:
        raise _exceptions.InvalidCategory(f'Argument catgeory must be one of: {default_preprocessor}')

    # prepare column argument
    if isinstance(columns, str):
        columns = [columns]
    split_cols = [idx for idx,cols in enumerate(columns) if isinstance(cols, list)]
    if len(split_cols)>0:
        for idx in split_cols:
            # join dataframe values
            combined = ','.join(columns[idx])
            df[combined] = df[columns[idx]].fillna('').agg(' '.join, axis=1)
            # adjust columns argument
            columns[idx] = combined

    # check presence of columns
    columns = pd.Series(columns)
    missing = columns[~columns.isin(df)]
    if any(missing):
        raise _exceptions.MissingColumn(f'Argument columns not in DataFrame: {missing.tolist()}')

    # prepare multiple columns by pivoting into a single column
    values = df[columns].stack()

    # preprocess values by category type
    if preprocessor=='default':
        values = default_preprocessor[category](values)
    else:
        values = preprocessor(values)

    return values


def find_related(category, values, kneighbors, threshold, analyzer):

    # check kneighbors argument
    if not isinstance(kneighbors, int) or kneighbors<0:
        raise _exceptions.KneighborsRange(f'Argument kneighbors must be a positive integer.')

    # check threshold argument
    if threshold<=0 or threshold>1:
        raise _exceptions.ThresholdRange(f'Argument threshold must be >0 and <=1.')

    # identify exact matches
    related = values[values.duplicated(keep=False) & values.notna()]
    related = related.groupby(related)
    related = related.ngroup()
    related = related.reset_index()
    related.columns = ['index','column','id_exact']
    
    # compare similarity
    if threshold==1:
        related['id_similar'] = pd.NA
        related['id'] = related['id_exact']
        similar = pd.DataFrame(columns=['score','id_similar','index','column','index_similar','column_similar'])
    else:
        # remove duplicates to lower computations needed for similar matching
        unique = values.drop_duplicates()

        # remove missing values that were completely removed during preprocessing
        unique = unique.dropna()

        # create TF-IDF matrix
        if analyzer=='default':
            analyzer = default_analyzer[category]
        vectorizer = TfidfVectorizer(
            # create features using words or characters
            analyzer=analyzer,
            # require 1 alphanumeric character instead of 2 to identify a word 
            token_pattern=r'(?u)\b\w+\b',
            # performed during preprocessing
            lowercase=False, 
            # removed during preprocessing
            stop_words=None
        )
        tfidf = vectorizer.fit_transform(unique.to_list())

        # create TF-IDF index relation to original data index
        tfidf_index = unique.index.to_frame(index=False, name=['index','column'])
        tfidf_index.name = 'tfidf_index'

        # initialize non-metric space libary for sparse matrix searching
        index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
        index.addDataPointBatch(tfidf)
        index.createIndex()

        # find nearest neighbors for the TF-IDF matrix
        similar = index.knnQueryBatch(tfidf, k=kneighbors, num_threads=4)
        similar = pd.DataFrame(similar, columns=['other_index','score'], index=tfidf_index.index)
        similar.index.name = 'tfidf_index'
        similar = similar.apply(pd.Series.explode)
        similar = similar.reset_index()
        similar['score'] = similar['score']*-1

        # check kneighbors and threshold comination potentially excluding matches
        # TODO: issue warning or raise exception if possible
        # last = similar.groupby('tfidf_index').agg({'score': min})
        # if any(last['score']>threshold):
        #     raise _exceptions.KneighborsThreshold(f'Similar matches excluded with kneighbors={kneighbors} and threshold={threshold}')

        # ignore matches to the same value
        similar = similar[similar['tfidf_index']!=similar['other_index']]

        # apply threshold
        similar = similar[similar['score']>=threshold]     

        # assign id to similarly connected components
        connected = nx.from_pandas_edgelist(similar, source='tfidf_index', target='other_index')
        connected = pd.DataFrame({'tfidf_index': list(nx.connected_components(connected))})
        connected.index.name = 'id_similar'
        connected = connected.explode('tfidf_index').reset_index()
        similar = similar.merge(connected, left_on='tfidf_index', right_on='tfidf_index')

        # replace tfidf_index with original unique data index
        similar = similar.merge(tfidf_index, left_on='tfidf_index', right_index=True)
        similar = similar.merge(tfidf_index, left_on='other_index', right_index=True, suffixes=('','_similar'))
        similar = similar.drop(columns=['tfidf_index','other_index'])

        # develop overall id by assuming the same id_exact for each group of id_similar (id_exact used is most frequent overall)
        related = related.merge(similar[['index','column','id_similar']], on=['index','column'], how='left')
        related['id_similar'] = related['id_similar'].astype('Int64')
        frequent = related['id_exact'].value_counts().to_frame(name='count')
        frequent.index.name = 'id_exact'
        frequent = frequent.reset_index()
        frequent = frequent.merge(related[['id_exact','id_similar']], on='id_exact')
        frequent = frequent.sort_values('count', ascending=False)
        frequent = frequent[['id_exact','id_similar']]
        frequent = frequent.dropna(subset='id_similar')
        frequent = frequent.drop_duplicates('id_similar', keep='first')
        frequent = frequent.rename(columns={'id_exact': 'id'})
        related = related.merge(frequent, on='id_similar', how='left')
        related['id'] = related['id'].fillna(related['id_exact'])

    # format return datatypes the same for exact or similar
    related = related.astype({'index': 'int64', 'column': 'string', 'id_exact': 'int64', 'id_similar': 'Int64', 'id': 'int64'})
    similar = similar.astype({'score': 'float64', 'id_similar': 'int64', 'index': 'int64', 'column': 'string', 'index_similar': 'int64', 'column_similar': 'string'})

    # add caregory description to ids for traceability
    related.columns = related.columns.str.replace(r'^id', f'{category}_id', regex=True)
    similar.columns = similar.columns.str.replace(r'^id', f'{category}_id', regex=True)

    # set index as the frames index
    related = related.set_index('index')
    similar = similar.set_index('index')

    return related, similar


def common_features(relationship):

    df_feature = []
    for category,related in relationship.items():

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


def assign_id(graph, id_name):

    df_id = pd.DataFrame({'index': list(nx.connected_components(graph))})
    df_id[id_name] = range(0, len(df_id))
    df_id = df_id.explode('index')

    return df_id


# def _assign_name(self, df_id, name_id, name_out):

#     common = self._processed['name'].reset_index()
#     common = common.drop(columns='column')
#     common = df_id.merge(common, on='index')
#     common = common.value_counts([name_id,'name']).reset_index()
#     common = common.drop(columns=0)
#     common = common.drop_duplicates(subset=name_id)
#     common = common.rename(columns={'name': name_out})
#     df_id = df_id.merge(common, on=name_id)

#     # set index of id dataframe as original source index
#     df_id = df_id.set_index('index')

#     return df_id