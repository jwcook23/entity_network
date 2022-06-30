'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations

import pandas as pd

from entity_network import _index, _prepare, _compare, _helpers

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _index.unique(df, df2)

        self.related = {}
        self.similar = {}
        # self.network_feature = None
        # self.network_id = None
        # self.entity_feature = None
        # self.entity_id = None
        self.processed = {}
        self.timer = pd.DataFrame(columns=['caller','file','method','category','time_seconds'])


    def compare(self, category, columns, kneighbors:int=10, threshold:float=1, text_comparer='default', text_cleaner='default'):

        # combine split columns, flatten into single
        print(f'flattening {columns}')
        tstart = time()
        self.processed[category] = _prepare.flatten(self._df, columns)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'flatten', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # clean column text
        if text_cleaner is not None:
            print(f'cleaning {category}')
            tstart = time()
            self.processed[category] = _prepare.clean(self.processed[category], category, text_cleaner)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'clean', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # set sources for tracability when comparing multiple categories
        self.processed[category].index.names = ('index', 'column')
        self.processed[category].name = category

        # ignore values the processor completely removed
        # self.processed[category] = self.processed[category].dropna()

        # compare values on similarity threshold
        print(f'comparing {columns}')
        tstart = time()
        self.related[category], self.similar[category] = _compare.match(category, self.processed[category], kneighbors, threshold, text_comparer, self._index_mask)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # add original index to processed values
        self.processed[category] = self.processed[category].reset_index()
        self.processed[category] = _index.original(self.processed[category], self._index_mask)

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)


    def network(self):

        print('forming network')

        # # determine network by matching features
        # tstart = time()
        # network_feature = _helpers.common_features(self.related)
        # self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__common_features', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # # create network graph
        # tstart = time()
        # self.network_graph = _helpers.series_graph(network_feature['index'])
        # self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__series_graph', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # assign network_id to connected records
        # tstart = time()
        # network_id = _helpers.assign_id(self.network_graph, 'network_id')
        # self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__assign_id', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # create network by using relationships
        tstart = time()
        network_id, network_feature = _helpers.assign_group(self.related, self._df.index)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_helpers', 'assign_group', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)
        
        self.related['phone'].head(2)
        self.related['email'].head(2)
        network_id.head(2)
        network_feature.head(2)

        # add original index
        tstart = time()
        network_id, network_feature = _index.related(network_id, network_feature, self._index_mask)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_index', 'original', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return network_id, network_feature


    def index_comparison(self, category, index_df: list = None, index_df2: list = None):

        # check if class initialized with one or two dataframes
        two_dfs = self._index_mask['df2'] is not None

        # define columns based on number of initialized dataframes
        if two_dfs:
            cols_processed = ['df_index','df2_index',category]
            cols_score = ['df_index', 'df_index_similar', 'df2_index','df2_index_similar', 'score']
            cols_exact = ['df_index', 'df2_index']
        else:
            cols_processed = ['df_index', category]
            cols_score = ['df_index', 'df_index_similar', 'score']
            cols_exact = ['df_index']

        # select processed text and similarity score for category
        processed = self.processed[category][cols_processed]
        score = self.similar[category][cols_score]

        # include exact matches
        id_exact = f'{category}_id_exact'
        exact = self.related[category][cols_exact+[id_exact]]
        if exact[id_exact].notna().any():
            list_notna = lambda l: [x for x in l if pd.notna(x)]
            list_combo = lambda x: list(combinations(x,2)) if len(x)>1 else ([(x[0],pd.NA)] if len(x)>0 else [(pd.NA, pd.NA)])
            # find combinations for exact match groups
            exact = exact.groupby(id_exact)
            exact = exact.agg({col: list_notna for col in cols_exact})
            # expand first df combos
            exact['df_index'] = exact['df_index'].apply(list_combo)
            exact = exact.explode('df_index')
            exact[['df_index','df_index_similar']] = exact['df_index'].to_list()
            exact[['df_index','df_index_similar']] = exact[['df_index','df_index_similar']].astype('Int64')
            # expand second df combos
            if two_dfs:
                exact['df2_index'] = exact['df2_index'].apply(list_combo)
                exact = exact.explode('df2_index')
                exact[['df2_index','df2_index_similar']] = exact['df2_index'].to_list()
                exact[['df2_index','df2_index_similar']] = exact[['df2_index','df2_index_similar']].astype('Int64')
            # add exact values
            score = pd.concat([score, exact])
            score['score'] = score['score'].fillna(1.0)
            # include related similarity for exact matches between two dataframes
            if two_dfs:
                related = exact[['df_index','df2_index','df2_index_similar']]
                related = related.rename(columns={'df_index': 'df_index_similar'})
                related = related.merge(score.drop(columns=['df2_index','df2_index_similar']), on='df_index_similar')
                related['df_index_similar'] = pd.NA
                related['df_index_similar'] = related['df_index_similar'].astype('Int64')
                score = pd.concat([score, related], ignore_index=True)

                related = exact[['df_index','df_index_similar']]
                related = related.merge(score[score['score']<1.0].drop(columns='df_index_similar'), on='df_index')
                related['df_index'] = pd.NA
                related['df_index'] = related['df_index'].astype('Int64')
                score = pd.concat([score, related], ignore_index=True)


        # error check index inputs
        if index_df is not None:
            check = pd.Series(index_df)
            missing = ~check.isin(processed['df_index'])
            if any(missing):
                raise RuntimeError(f'index_df provided is not a valid index: {list(check[missing])}')
        if index_df2 is not None:
            check = pd.Series(index_df2)
            missing = ~check.isin(processed['df2_index'])
            if any(missing):
                raise RuntimeError(f'index_df2 provided is not a valid index: {list(check[missing])}')

        # select indices by parameters given
        if index_df is not None and index_df2 is not None:
            comparison = score[
                (score['df_index'].isin(index_df) | score['df_index_similar'].isin(index_df)) &
                (score['df2_index'].isin(index_df2) | score['df2_index_similar'].isin(index_df2))
            ]
        elif index_df is not None:
            comparison = score[score['df_index'].isin(index_df) | score['df_index_similar'].isin(index_df)]
        elif index_df2 is not None:
            comparison = score[score['df2_index'].isin(index_df2) | score['df2_index_similar'].isin(index_df2)]

        # remove mirrored matches where other_index, tfidf_index is the reverse of tfidf_index, other_index
        first = ~comparison[['df_index','df_index_similar']].apply(frozenset, axis=1).duplicated() 
        comparison = comparison[first]

        # add processed values compared for df_index, df_index_similar
        values = processed[['df_index', category]].dropna(subset='df_index')
        comparison = comparison.merge(
            values.rename(columns={category: f'df_{category}'}),
            on='df_index', how='left'
        )
        comparison = comparison.merge(
            values.rename(columns={'df_index': 'df_index_similar', category: f'df_{category}_similar'}),
            on='df_index_similar', how='left'
        )
        # add processed values compared for df2_index, df2_index_similar
        if two_dfs:
            values = processed[['df2_index', category]].dropna(subset='df2_index')
            comparison = comparison.merge(
                values.rename(columns={category: f'df2_{category}'}),
                on='df2_index', how='left'
            )
            comparison = comparison.merge(
                values.rename(columns={'df2_index': 'df2_index_similar', category: f'df2_{category}_similar'}),
                on='df2_index_similar', how='left'
            )

        # # add similar values from first df
        # similar = processed[['df_index', category]]
        # similar = similar.dropna(subset='df_index')
        # similar = similar.rename(columns={'df_index': 'df_index_similar'})
        # comparison = comparison.merge(similar, on='df_index_similar', suffixes=('','_df_similar'), how='left')

        # # add similar values from second df
        # if two_dfs:
        #     similar = processed[['df2_index', category]]
        #     similar = similar.dropna(subset='df2_index')
        #     similar = similar.rename(columns={'df2_index': 'df2_index_similar'})
        #     comparison = comparison.merge(similar, on='df2_index_similar', suffixes=('','_df2_similar'), how='left')

        # sort by first appearing index
        if index_df2 is not None:
            comparison = comparison.sort_values('df2_index')
        else:
            comparison = comparison.sort_values('df_index')

        # for equality testing, reset index that doesn't carry meaning
        comparison = comparison.reset_index(drop=True)

        return comparison