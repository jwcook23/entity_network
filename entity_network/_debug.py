import pandas as pd

from entity_network._find_difference import term_difference

def first_df(score, processed, category, exact):
    
    # create a copy to preserve the original df
    processed = processed['df'].copy()
    
    # track which df the processed value originated from
    processed.name = f'{category}_df_value'

    # include exact matches within the first df
    if exact is not None:
        fill = processed.reset_index(level=1, drop=True)
        fill = exact.merge(fill, left_index=True, right_index=True)
        fill = fill.drop(columns='id')
        fill = fill.set_index(keys=['node','column'])
        fill = fill[processed.name]
        processed = pd.concat([processed, fill], ignore_index=False)
    
    # combine processed values with their similarity score
    score = score.merge(processed, how='left', on=['node','column'])

    return score

def similar_values(score, processed, category):

    if processed['df2'] is None:
        processed = processed['df'].copy()
        processed.name = f'{category}_df_similar_value'
        score = score.merge(processed, how='left', left_on=['node_similar','column'], right_on=['node','column'])
    else:
        processed = processed['df2'].copy().reset_index()
        processed = processed.rename(columns={'node': 'node_similar', category: f'{category}_df2_similar_value', 'column': 'column_df2'})
        score = score.merge(processed, how='left', on='node_similar')

    return score


def cluster_edge(score, cluster_edge_limit):

    # find values closest to not being included
    in_cluster = score[score['threshold']].sort_values(by='score', ascending=True)
    in_cluster = in_cluster.head(cluster_edge_limit)
    
    # find records closest to belonging to a cluster
    out_cluster = score[~score['threshold']].sort_values(by='score', ascending=False)
    out_cluster = out_cluster.head(cluster_edge_limit)

    return in_cluster, out_cluster


def record_difference(score, in_cluster, out_cluster, category):

    compare = score.columns[score.columns.str.endswith('_value')]
    diff = compare.str.replace('_value$', '_diff', regex=True)
    in_cluster[[diff[0],diff[1]]] = in_cluster[compare].apply(term_difference[category], axis=1, result_type='expand')
    out_cluster[[diff[0],diff[1]]] = out_cluster[compare].apply(term_difference[category], axis=1, result_type='expand')

    return in_cluster, out_cluster