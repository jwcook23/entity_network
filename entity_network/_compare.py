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
        similar = None
    else:
        # remove duplicates to lower computations needed for similar matching
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
        similar = index.knnQueryBatch(tfidf, k=kneighbors, num_threads=4)
        similar = pd.DataFrame(similar, columns=['other_index','score'], index=tfidf_index.index)
        similar.index.name = 'tfidf_index'
        similar = similar.apply(pd.Series.explode)
        similar['other_index'] = similar['other_index'].astype('int64')
        similar['score'] = similar['score'].astype('float64')
        similar = similar.reset_index()
        similar['score'] = similar['score']*-1

        # check kneighbors and threshold comination potentially excluding matches
        # TODO: issue warning or raise exception if possible
        # last = similar.groupby('tfidf_index').agg({'score': min})
        # if any(last['score']>threshold):
        #     raise _exceptions.KneighborsThreshold(f'Similar matches excluded with kneighbors={kneighbors} and threshold={threshold}')

        # ignore matches to the same value
        similar = similar[similar['tfidf_index']!=similar['other_index']]   

        # # remove mirrored matches where other_index, tfidf_index is the reverse of tfidf_index, other_index
        # ordered = similar[['tfidf_index','other_index']].values
        # ordered.sort(axis=1)
        # similar[['tfidf_index','other_index']] = ordered
        # similar = similar.drop_duplicates(subset=['tfidf_index','other_index'])

        # replace tfidf_index with original unique data index
        similar = similar.merge(tfidf_index, left_on='tfidf_index', right_index=True)
        similar = similar.merge(tfidf_index, left_on='other_index', right_index=True, suffixes=('','_similar'))
        similar = similar.drop(columns=['tfidf_index','other_index'])

        # apply threshold criteria
        similar['threshold'] = similar['score']>=threshold

        # place most similar values first for first dataframe
        similar = similar.sort_values(by=['index', 'score'], ascending=[True, False])

        # assign id to similarly connected components
        connected = nx.from_pandas_edgelist(similar[similar['threshold']], source='index', target='index_similar')
        connected = pd.DataFrame({'index': list(nx.connected_components(connected))})
        connected.index.name = 'id_similar'
        connected = connected.explode('index').reset_index()
        connected['threshold'] = True
        similar = similar.merge(connected, on=['index','threshold'], how='left')
        similar['id_similar'] = similar['id_similar'].astype('Int64')

        # add similar into exact matches
        related = pd.concat([related, similar.loc[similar['threshold'], ['index','column','id_similar']]], ignore_index=True)
        related[['id_exact','id_similar']] = related[['id_exact','id_similar']].astype('Int64')

        # develop an overall id
        related['temp_id'] = related.groupby(['id_exact','id_similar'], dropna=False).ngroup()
        combine = related.groupby('index')
        combine = combine.agg({'temp_id': list})
        combine['id'] = combine['temp_id'].apply(lambda x: x[0])
        combine = combine.explode('temp_id')
        combine = combine.drop_duplicates(keep='first', subset='temp_id')
        related = related.merge(combine, on='temp_id')
        related = related.drop(columns='temp_id')

    # format final return data types
    # TODO: change data types earlier
    related = _index.original(related, index_mask)
    related = related.astype({'index': 'int64', 'column': 'string', 'id_exact': 'Int64', 'id_similar': 'Int64', 'id': 'int64'})
    related = related.set_index('index')
    if similar is not None:
        similar = _index.original(similar, index_mask)
        similar = _index.original(similar, index_mask, index_name='index_similar')
        similar = similar.astype({'score': 'float64', 'id_similar': 'Int64', 'index': 'int64', 'column': 'string', 'index_similar': 'int64', 'column_similar': 'string'})
        similar = similar.set_index('index')

    return related, similar

