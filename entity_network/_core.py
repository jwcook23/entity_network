import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
import networkx as nx

from entity_network import clean_text, _exceptions

category_processors = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}
category_analyzer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def prepare_values(df, category, columns):

    # check allowed category argument
    if not category in category_processors:
        raise _exceptions.InvalidCategory(f'Argument catgeory must be one of: {category_processors}')

    # check and prepare column argument
    if isinstance(columns, str):
        columns = [columns]
    columns = pd.Series(columns)
    missing = columns[~columns.isin(df)]
    if any(missing):
        raise _exceptions.MissingColumn(f'Argument columns not in DataFrame: {missing.tolist()}')

    # prepare multiple columns by pivoting into a single column
    values = df[columns].stack()

    # preprocess values by category type
    values = category_processors[category](values)

    return values


def find_related(category, values, kneighbors, threshold):

    # check kneighbors argument
    if not isinstance(kneighbors, int) or kneighbors<0:
        raise _exceptions.KneighborsRange(f'Argument kneighbors must be a positive integer.')

    # check threshold argument
    if threshold<=0 or threshold>1:
        raise _exceptions.ThresholdRange(f'Argument threshold must be >0 and <=1.')

    # identify exact matches
    related = values.groupby(values).ngroup()
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

        # create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            # create features using words or characters
            analyzer=category_analyzer[category],
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
        last = similar.groupby('tfidf_index').agg({'score': min})
        if any(last['score']>threshold):
            raise _exceptions.KneighborsThreshold(f'Similar matches excluded with kneighbors={kneighbors} and threshold={threshold}')

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
    similar = similar.astype({'score': 'float64', 'id_similar': 'int64', 'index': 'int64', 'column': 'string', 'index_similar': 'int64', 'column_similar': 'int64'})

    # add caregory description to ids for traceability
    related.columns = related.columns.str.replace(r'^id', f'{category}_id', regex=True)
    similar.columns = similar.columns.str.replace(r'^id', f'{category}_id', regex=True)

    return related, similar
