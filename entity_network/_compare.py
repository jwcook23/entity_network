import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
import networkx as nx

from entity_network import _index, _exceptions

default_text_comparer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def match(category, values, kneighbors, threshold, text_comparer, index_mask):

    id_category = f'{category}_id'

    # check kneighbors argument
    if not isinstance(kneighbors, int) or kneighbors<0:
        raise _exceptions.KneighborsRange(f'Argument kneighbors must be a positive integer.')

    # check threshold argument
    if threshold<=0 or threshold>1:
        raise _exceptions.ThresholdRange(f'Argument threshold must be >0 and <=1.')

    # identify exact matches
    related_feature = values[values.duplicated(keep=False) & values.notna()]
    related_feature = related_feature.groupby(related_feature)
    related_feature = related_feature.ngroup()
    related_feature = related_feature.reset_index()
    related_feature.columns = ['index','column','id_exact']
    
    # compare similarity
    if threshold==1:
        related_feature['id_similar'] = pd.NA
        related_feature[id_category] = related_feature['id_exact']
        similar_feature = None
    else:
        # remove duplicates to lower computations needed for similar_feature matching
        unique = values.drop_duplicates()

        # remove missing values that were completely removed during preprocessing
        unique = unique.dropna()

        # create TF-IDF matrix
        if text_comparer=='default':
            text_comparer = default_text_comparer[category]
        vectorizer = TfidfVectorizer(
            # create features using words or characters
            analyzer=text_comparer,
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
        similar_feature = index.knnQueryBatch(tfidf, k=kneighbors, num_threads=4)
        similar_feature = pd.DataFrame(similar_feature, columns=['other_index','score'], index=tfidf_index.index)
        similar_feature.index.name = 'tfidf_index'
        similar_feature = similar_feature.apply(pd.Series.explode)
        similar_feature['other_index'] = similar_feature['other_index'].astype('int64')
        similar_feature['score'] = similar_feature['score'].astype('float64')
        similar_feature = similar_feature.reset_index()
        similar_feature['score'] = similar_feature['score']*-1

        # check kneighbors and threshold comination potentially excluding matches
        # TODO: issue warning or raise exception if possible
        # last = similar_feature.groupby('tfidf_index').agg({'score': min})
        # if any(last['score']>threshold):
        #     raise _exceptions.KneighborsThreshold(f'similar_feature matches excluded with kneighbors={kneighbors} and threshold={threshold}')

        # ignore matches to the same value
        similar_feature = similar_feature[similar_feature['tfidf_index']!=similar_feature['other_index']]   

        # # remove mirrored matches where other_index, tfidf_index is the reverse of tfidf_index, other_index
        # ordered = similar_feature[['tfidf_index','other_index']].values
        # ordered.sort(axis=1)
        # similar_feature[['tfidf_index','other_index']] = ordered
        # similar_feature = similar_feature.drop_duplicates(subset=['tfidf_index','other_index'])

        # replace tfidf_index with original unique data index
        similar_feature = similar_feature.merge(tfidf_index, left_on='tfidf_index', right_index=True)
        similar_feature = similar_feature.merge(tfidf_index, left_on='other_index', right_index=True, suffixes=('','_similar'))
        similar_feature = similar_feature.drop(columns=['tfidf_index','other_index'])

        # apply threshold criteria
        similar_feature['threshold'] = similar_feature['score']>=threshold

        # place most similar_feature values first for first dataframe
        similar_feature = similar_feature.sort_values(by=['index', 'score'], ascending=[True, False])

        # assign id to similarly connected components
        connected = nx.from_pandas_edgelist(similar_feature[similar_feature['threshold']], source='index', target='index_similar')
        connected = pd.DataFrame({'index': list(nx.connected_components(connected))})
        connected.index.name = 'id_similar'
        connected = connected.explode('index').reset_index()
        connected['threshold'] = True
        similar_feature = similar_feature.merge(connected, on=['index','threshold'], how='left')
        similar_feature['id_similar'] = similar_feature['id_similar'].astype('Int64')

        # add similar_feature into exact matches
        related_feature = pd.concat([related_feature, similar_feature.loc[similar_feature['threshold'], ['index','column','id_similar']]], ignore_index=True)
        related_feature[['id_exact','id_similar']] = related_feature[['id_exact','id_similar']].astype('Int64')

        # develop an overall id
        related_feature['temp_id'] = related_feature.groupby(['id_exact','id_similar'], dropna=False).ngroup()
        combine = related_feature.groupby('index')
        combine = combine.agg({'temp_id': list})
        combine[id_category] = combine['temp_id'].apply(lambda x: x[0])
        combine = combine.explode('temp_id')
        combine = combine.drop_duplicates(keep='first', subset='temp_id')
        related_feature = related_feature.merge(combine, on='temp_id')
        related_feature = related_feature.drop(columns='temp_id')

    # format final return data types
    # TODO: change data types earlier
    related_feature = _index.original(related_feature, index_mask)
    related_feature = related_feature.astype({'index': 'int64', 'column': 'string', 'id_exact': 'Int64', 'id_similar': 'Int64', id_category: 'int64'})
    related_feature = related_feature.set_index('index')
    if similar_feature is not None:
        similar_feature = _index.original(similar_feature, index_mask)
        similar_feature = _index.original(similar_feature, index_mask, index_name='index_similar')
        similar_feature = similar_feature.astype({'score': 'float64', 'id_similar': 'Int64', 'index': 'int64', 'column': 'string', 'index_similar': 'int64', 'column_similar': 'string'})
        similar_feature = similar_feature.set_index('index')

    # remove features that only self-match (can occur since multiple columns may be stacked and compared)
    remove_index = related_feature.reset_index().groupby(id_category).agg({'index': 'nunique'})
    remove_index = remove_index[remove_index['index']==1].index
    remove_index = related_feature.index[related_feature[id_category].isin(remove_index)]
    related_feature = related_feature[~related_feature.index.isin(remove_index)]
    if similar_feature is not None:
        similar_feature = similar_feature[~similar_feature.index.isin(remove_index)]

    return related_feature, similar_feature

