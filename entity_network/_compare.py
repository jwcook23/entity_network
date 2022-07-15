import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import usaddress

from entity_network import _index

default_text_comparer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def exact_match(values):

    related_feature = values[values.duplicated(keep=False) & values.notna()]
    related_feature = related_feature.groupby(related_feature)
    related_feature = related_feature.ngroup()
    related_feature = related_feature.reset_index()
    related_feature.columns = ['node','column','id_exact']

    return related_feature

def create_tfidf(category, values, text_comparer):

    # remove duplicates to lower computations needed for similar_feature matching
    unique = values.drop_duplicates()

    # remove missing values that were completely removed during preprocessing
    unique = unique.dropna()

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
    similar_feature = unique.index.to_frame(index=False, name=['node','column'])
    similar_feature.name = 'tfidf_index'

    return tfidf, similar_feature

def similar_match(tfidf, kneighbors):

    # initialize non-metric space libary for sparse matrix searching
    index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
    index.addDataPointBatch(tfidf)
    index.createIndex()

    # find similar features matching above
    # TODO: ignore first half since it will be repeated information?
    similar_score = index.knnQueryBatch(tfidf, k=kneighbors, num_threads=4)

    return similar_score

def similar_id(similar_score, similar_feature, threshold):

    # assign a group id based on connected components above threshold
    n = similar_feature.index.max()+1
    # form list of lists sparse matrix incrementally
    graph = lil_matrix((n, n), dtype=int)
    for comparison in similar_score:
        # invert score for negative dot product
        score = comparison[1]*-1
        # assign matrix values that meet threshold
        idx = comparison[0][score>=threshold]
        graph[idx[0], idx] = 1
    # convert to compressed sparse row matrix for better computation
    graph = graph.tocsr()
    # find connected components to assign an id
    _, labels = connected_components(graph, directed=False, return_labels=True)
    similar_feature['id_similar'] = labels

    return similar_feature


def expand_score(similar_score, similar_feature, threshold):

    similar_score = pd.DataFrame(similar_score, columns=['tfidf_similar','score'], index=similar_feature.index)
    similar_score.index.name = 'tfidf_index'
    similar_score = similar_score.apply(pd.Series.explode)
    similar_score = similar_score[similar_score.index!=similar_score['tfidf_similar']]
    similar_score['tfidf_similar'] = similar_score['tfidf_similar'].astype('int64')
    similar_score['score'] = similar_score['score'].astype('float64')
    similar_score['score'] = similar_score['score']*-1
    similar_score['threshold'] = similar_score['score']>=threshold
    similar_score = similar_score.merge(similar_feature[['column','id_similar']], left_on='tfidf_similar', right_index=True)
    similar_score = similar_score.sort_values(by=['id_similar', 'score'], ascending=[True,False])

    # convert similar_score indexing from tfidf back to original
    similar_score.index = similar_feature.loc[similar_score.index,'node']
    similar_score['tfidf_similar'] = similar_feature.loc[similar_score['tfidf_similar'],'node'].values
    similar_score = similar_score.rename(columns={'tfidf_similar': 'node_similar'})

    return similar_score

def combined_id(related_feature, similar_feature, id_category):

    # only retain values similar to another
    multiple = similar_feature.groupby('id_similar').size()
    multiple = multiple[multiple>1]
    similar_feature = similar_feature[similar_feature['id_similar'].isin(multiple.index)]

    # add similar_feature into exact matches
    related_feature = pd.concat([related_feature, similar_feature], ignore_index=True)
    related_feature[['id_exact','id_similar']] = related_feature[['id_exact','id_similar']].astype('Int64')

    # develop an overall id based on both similar and exact id
    related_feature['temp_id'] = related_feature.groupby(['id_exact','id_similar'], dropna=False).ngroup()
    combine = related_feature.groupby('node')
    combine = combine.agg({'temp_id': list})
    combine[id_category] = combine['temp_id'].apply(lambda x: x[0])
    combine = combine.explode('temp_id')
    combine = combine.drop_duplicates(keep='first', subset='temp_id')
    related_feature = related_feature.merge(combine, on='temp_id')
    related_feature = related_feature.drop(columns='temp_id')

    return related_feature, similar_feature

def remove_self_matches(related_feature, similar_score, id_category):

    remove_index = related_feature.reset_index().groupby(id_category).agg({'node': 'nunique'})
    remove_index = remove_index[remove_index['node']==1].index
    remove_index = related_feature.index[related_feature[id_category].isin(remove_index)]
    related_feature = related_feature[~related_feature.index.isin(remove_index)]
    if similar_score is not None:
        similar_score = similar_score[~similar_score.index.isin(remove_index)]

    return related_feature, similar_score

def translate_index(related_feature, similar_score, index_mask, id_category):

    related_feature = _index.original(related_feature, index_mask)
    related_feature = related_feature.astype({'node': 'int64', 'column': 'string', 'id_exact': 'Int64', 'id_similar': 'Int64', id_category: 'int64'})
    related_feature = related_feature.set_index('node')
    if similar_score is not None:
        similar_score = similar_score.reset_index()
        similar_score = _index.original(similar_score, index_mask, index_name='node')
        similar_score = _index.original(similar_score, index_mask, index_name='node_similar')
        similar_score = similar_score.set_index(['node','node_similar'])

    return related_feature, similar_score

# def match(category, values, kneighbors, threshold, text_comparer, index_mask):

#     id_category = f'{category}_id'

#     related_feature = values[values.duplicated(keep=False) & values.notna()]
#     related_feature = related_feature.groupby(related_feature)
#     related_feature = related_feature.ngroup()
#     related_feature = related_feature.reset_index()
#     related_feature.columns = ['node','column','id_exact']
    
#     # compare similarity
#     if threshold==1:
#         related_feature['id_similar'] = pd.NA
#         related_feature[id_category] = related_feature['id_exact']
#         similar_score = None
#     else:
#         # remove duplicates to lower computations needed for similar_feature matching
#         unique = values.drop_duplicates()

#         # remove missing values that were completely removed during preprocessing
#         unique = unique.dropna()

#         # create TF-IDF matrix
#         if text_comparer=='default':
#             text_comparer = default_text_comparer[category]
#         vectorizer = TfidfVectorizer(
#             # create features using words or characters
#             analyzer=text_comparer,
#             # require 1 alphanumeric character instead of 2 to identify a word 
#             token_pattern=r'(?u)\b\w+\b',
#             # performed during preprocessing
#             lowercase=False, 
#             # removed during preprocessing
#             stop_words=None
#         )
#         tfidf = vectorizer.fit_transform(unique.to_list())

#         # create TF-IDF index relation to original data index
#         similar_feature = unique.index.to_frame(index=False, name=['node','column'])
#         similar_feature.name = 'tfidf_index'

#         # initialize non-metric space libary for sparse matrix searching
#         index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
#         index.addDataPointBatch(tfidf)
#         index.createIndex()

#         # find similar features matching above
#         # TODO: ignore first half since it will be repeated information?
#         similar_score = index.knnQueryBatch(tfidf, k=kneighbors, num_threads=4)

#         # assign a group id based on connected components above threshold
#         n = similar_feature.index.max()+1
#         # form list of lists sparse matrix incrementally
#         graph = lil_matrix((n, n), dtype=int)
#         for comparison in similar_score:
#             # invert score for negative dot product
#             score = comparison[1]*-1
#             # assign matrix values that meet threshold
#             idx = comparison[0][score>=threshold]
#             graph[idx[0], idx] = 1
#         # convert to compressed sparse row matrix for better computation
#         graph = graph.tocsr()
#         # find connected components to assign an id
#         _, labels = connected_components(graph, directed=False, return_labels=True)
#         similar_feature['id_similar'] = labels

#         # expand similar feature score and ignore self matches
#         similar_score = pd.DataFrame(similar_score, columns=['tfidf_similar','score'], index=similar_feature.index)
#         similar_score.index.name = 'tfidf_index'
#         similar_score = similar_score.apply(pd.Series.explode)
#         similar_score = similar_score[similar_score.index!=similar_score['tfidf_similar']]
#         similar_score['tfidf_similar'] = similar_score['tfidf_similar'].astype('int64')
#         similar_score['score'] = similar_score['score'].astype('float64')
#         similar_score['score'] = similar_score['score']*-1
#         similar_score['threshold'] = similar_score['score']>=threshold
#         similar_score = similar_score.merge(similar_feature[['column','id_similar']], left_on='tfidf_similar', right_index=True)
#         similar_score = similar_score.sort_values(by=['id_similar', 'score'], ascending=[True,False])

#         # convert similar_score indexing from tfidf back to original
#         similar_score.index = similar_feature.loc[similar_score.index,'node']
#         similar_score['tfidf_similar'] = similar_feature.loc[similar_score['tfidf_similar'],'node'].values
#         similar_score = similar_score.rename(columns={'tfidf_similar': 'node_similar'})

#         # only retain values similar to another
#         multiple = similar_feature.groupby('id_similar').size()
#         multiple = multiple[multiple>1]
#         similar_feature = similar_feature[similar_feature['id_similar'].isin(multiple.index)]

#         # add similar_feature into exact matches
#         related_feature = pd.concat([related_feature, similar_feature], ignore_index=True)
#         related_feature[['id_exact','id_similar']] = related_feature[['id_exact','id_similar']].astype('Int64')

#         # develop an overall id based on both similar and exact id
#         related_feature['temp_id'] = related_feature.groupby(['id_exact','id_similar'], dropna=False).ngroup()
#         combine = related_feature.groupby('node')
#         combine = combine.agg({'temp_id': list})
#         combine[id_category] = combine['temp_id'].apply(lambda x: x[0])
#         combine = combine.explode('temp_id')
#         combine = combine.drop_duplicates(keep='first', subset='temp_id')
#         related_feature = related_feature.merge(combine, on='temp_id')
#         related_feature = related_feature.drop(columns='temp_id')

#     # remove features that only self-match (can occur since multiple columns may be stacked and compared)
#     remove_index = related_feature.reset_index().groupby(id_category).agg({'node': 'nunique'})
#     remove_index = remove_index[remove_index['node']==1].index
#     remove_index = related_feature.index[related_feature[id_category].isin(remove_index)]
#     related_feature = related_feature[~related_feature.index.isin(remove_index)]
#     if similar_score is not None:
#         similar_score = similar_score[~similar_score.index.isin(remove_index)]

#     # translate to original input index and format final return data types
#     # TODO: change data types earlier
#     related_feature = _index.original(related_feature, index_mask)
#     related_feature = related_feature.astype({'node': 'int64', 'column': 'string', 'id_exact': 'Int64', 'id_similar': 'Int64', id_category: 'int64'})
#     related_feature = related_feature.set_index('node')
#     if similar_score is not None:
#         similar_score = similar_score.reset_index()
#         similar_score = _index.original(similar_score, index_mask, index_name='node')
#         similar_score = _index.original(similar_score, index_mask, index_name='node_similar')
#         similar_score = similar_score.set_index(['node','node_similar'])

#     return related_feature, similar_score

def address(values):

    values = values.dropna()

    # parse address components
    components0,_ = usaddress.tag(values[0])
    components0 = set(components0.items())

    components1,_ = usaddress.tag(values[1])
    components1 = set(components1.items())

    # determine difference in address components
    diff0 = components0-components1
    if len(diff0)==0:
        diff0 = None
    diff1 = components1-components0
    if len(diff1)==0:
        diff1 = None

    return diff0, diff1