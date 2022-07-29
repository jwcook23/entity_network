import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from entity_network import _index

default_text_comparer = {
    'name': 'char',
    'phone': 'char',
    'email': 'char',
    'email_domain': 'char',
    'address': 'word'
}

def exact_match(values):

    if values['df2'] is None:
        # compare values in single dataframe if that is all that is given
        values = values['df']
    else:
        # compare values in df and df2
        values = pd.concat([
            values['df'][values['df'].isin(values['df2'])], 
            values['df2'][values['df2'].isin(values['df'])]
        ])

    # label exact matches with an id
    related_feature = values[values.duplicated(keep=False) & values.notna()]
    related_feature = related_feature.groupby(related_feature)
    related_feature = related_feature.ngroup()
    related_feature = related_feature.reset_index()
    related_feature.columns = ['node','column','id_exact']

    return related_feature

def create_tfidf(category, values, text_comparer, related_feature):

    # remove duplicates and nulls to lower kneighbors parameter needed
    for frame in values.keys():
        if values[frame] is not None:
            # remove duplicates in the same dataframe or other frame identified in previous step
            # values[frame] = values[frame][~values[frame].index.get_level_values('node').isin(related_feature['node'])]
            values[frame] = values[frame].drop_duplicates()
            # remove missing values that were completely removed during preprocessing
            values[frame] = values[frame].dropna()

    # define vectorizer to transform text to numbers
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

    # create tfidf and a map between tfidf_index and nodes
    tfidf = {'df': None, 'df2': None}
    tfidf_index = {'df': None, 'df2': None}
    for frame, data in values.items():
        # skip df2 if not provided
        if data is None:
            continue
        # transform text to tfidf
        if frame=='df':
            tfidf[frame] = vectorizer.fit_transform(data.to_list())
        else:
            tfidf[frame] = vectorizer.transform(data.to_list())
        # create TF-IDF index relation to original data index
        tfidf_index[frame] = data.index.to_frame(index=False, name=['node','column'])
        tfidf_index[frame].index.name = 'tfidf_index'

    return tfidf, tfidf_index

def similar_match(tfidf, tfidf_index, kneighbors):

    # initialize non-metric space libary for sparse matrix searching
    index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 
    index.addDataPointBatch(tfidf['df'])
    index.createIndex()

    # find similar features matching above
    # TODO: ignore first half since it will be repeated information?
    if tfidf['df2'] is None:
        # find similar values for the first df in the first df
        neighbors = index.knnQueryBatch(tfidf['df'], k=kneighbors, num_threads=4)
    else:
        # find similar vlaues for the first df in the second df
        neighbors = index.knnQueryBatch(tfidf['df2'], k=kneighbors, num_threads=4)

    # replace tfidf_index with node index
    similar_score = {}
    for idx, comparison in enumerate(neighbors):
        if tfidf['df2'] is None:
            # lookup index in the single dataframe
            node_source = tfidf_index['df'].at[idx, 'node']
        else:
            # lookup node index in the second (smaller) dataframe
            node_source = tfidf_index['df2'].at[idx, 'node']
        # lookup node index in the first (larger) or single dataframe
        node_target = tfidf_index['df'].loc[comparison[0], 'node'].values
        # adjust score for negative dot product
        score = comparison[1]*-1
        # assign values using the first dataframe as the main dataframe
        similar_score[node_source] =  [node_target, score]

    return similar_score

def similar_id(similar_score, tfidf_index, threshold):

    # TODO: allow a single component difference, such as OccupancyIdentifier for address

    # determine node size needed
    if tfidf_index['df2'] is None:
        n = tfidf_index['df']['node'].max()+1
    else:
        n = tfidf_index['df2']['node'].max()+1
    
    # form list of lists sparse matrix incrementally
    graph = lil_matrix((n, n), dtype=int)
    for node_source, comparison in similar_score.items():
        # assign matrix values that meet threshold
        node_target = comparison[0][comparison[1]>=threshold]
        graph[node_source, node_target] = 1
    # convert to compressed sparse row matrix for better computation
    graph = graph.tocsr()

    # find connected components to assign an id
    _, labels = connected_components(graph, directed=False, return_labels=True)
    similar_feature = pd.DataFrame({
        'node': range(0, n),
        'id_similar': labels
    })

    # include source column info
    similar_feature = similar_feature.merge(tfidf_index['df'], on='node', how='left')
    if tfidf_index['df2'] is not None:
        similar_feature = similar_feature.merge(tfidf_index['df2'], on='node', how='left', suffixes=('','_df2'))
        similar_feature['column'] =  similar_feature['column'].fillna(similar_feature['column_df2'])
        similar_feature = similar_feature.drop(columns='column_df2')

    # return ids for the main dataframe only
    # similar_feature = similar_feature[similar_feature['node'].isin(tfidf_index['df']['node'])]

    return similar_feature

def expand_score(similar_score, similar_feature, threshold):

    # convert from dictionary to dataframe
    similar_score = pd.DataFrame.from_dict(similar_score, orient='index', columns=['node', 'score'])
    similar_score.index.name = 'node_similar'
    similar_score = similar_score.apply(pd.Series.explode)
    similar_score['node'] = similar_score['node'].astype('int64')
    similar_score['score'] = similar_score['score'].astype('float64')

    # ignore self matchings records
    similar_score = similar_score[similar_score.index!=similar_score['node']]

    # mark scores above threshold
    similar_score['threshold'] = similar_score['score']>=threshold

    # add column source and similar_id to score
    similar_score = similar_score.merge(similar_feature[['column','id_similar']], left_on='node', right_index=True)
    similar_score.index.name = 'node_similar'
    similar_score = similar_score.sort_values(by=['id_similar', 'score'], ascending=[True,False])

    return similar_score

def combined_id(related_feature, similar_feature, id_category):

    # only retain values similar to another
    multiple = similar_feature.groupby('id_similar').size()
    multiple = multiple[multiple>1]
    similar_feature = similar_feature[similar_feature['id_similar'].isin(multiple.index)]

    # add similar_feature into exact matches
    related_feature = related_feature.merge(similar_feature, on=['node','column'], how='outer', suffixes=('','_similar'))
    related_feature[['id_exact','id_similar']] = related_feature[['id_exact','id_similar']].astype('Int64')

    # develop an overall using both similar and exact ids
    # 1. assume the same exact id for the same similar id
    connected = related_feature[['id_similar','id_exact']].dropna()
    connected = connected.groupby('id_similar')
    connected = connected.agg({'id_exact': 'first'})
    connected = connected.rename(columns={'id_exact': id_category})
    related_feature = related_feature.merge(connected, left_on='id_similar', right_index=True, how='left')
    # 2. use exact id if not similar
    related_feature[id_category] = related_feature[id_category].fillna(related_feature['id_exact'])
    # 3. derive an id if only similar
    derive = related_feature.loc[related_feature[id_category].isna(),['id_similar']]
    derive = derive.drop_duplicates()
    seed = related_feature[id_category].max()+1
    if pd.isna(seed):
        seed = 0
    derive['temp_id'] = range(seed, len(derive)+seed)
    related_feature = related_feature.merge(derive, on='id_similar', how='left')
    related_feature['temp_id'] = related_feature['temp_id'].astype('Int64')
    related_feature[id_category] = related_feature[id_category].fillna(related_feature['temp_id'])
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
        similar_score = _index.original(similar_score, index_mask)
        # similar_score = _index.original(similar_score, index_mask, index_name='node_similar')
        similar_score = similar_score.set_index(['node','node_similar'])


    return related_feature, similar_score
