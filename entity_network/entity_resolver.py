'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations, product

import pandas as pd
import networkx as nx

from entity_network import _index, _prepare, _compare

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

        # determine network by matching features
        tstart = time()
        network_feature = self.__common_features()
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__common_features', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # create network graph
        tstart = time()
        self.network_graph = self.__series_graph(network_feature['index'])
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__series_graph', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # assign network_id to connected records
        tstart = time()
        network_id = self.__assign_id(self.network_graph, 'network_id')
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', 'self', '__assign_id', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # add original index
        tstart = time()
        network_id, network_feature = _index.related(network_id, network_feature, self._index_mask)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_index', 'original', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return network_id, network_feature


    def index_comparison(self, category, index_df: list = None, index_df2: list = None):

        # define columns based on number of initialized dataframes
        if self._index_mask['df2'] is None:
            cols_processed = ['df_index', category]
            cols_score = ['df_index', 'df_index_similar', 'score']
            cols_exact = ['df_index']
        else:
            cols_processed = ['df_index','df2_index',category]
            cols_score = ['df_index', 'df_index_similar', 'df2_index','df2_index_similar', 'score']
            cols_exact = ['df_index', 'df2_index']

        # select processed text and similarity score for category
        processed = self.processed[category][cols_processed]
        score = self.similar[category][cols_score]

        # include exact matches
        id_exact = f'{category}_id_exact'
        exact = self.related[category][cols_exact+[id_exact]]
        if exact[id_exact].notna().any():
            list_notna = lambda l: [x for x in l if pd.notna(x)]
            list_combo = lambda x: list(combinations(x,2)) if len(x)>1 else ([(x[0],)] if len(x)>0 else [tuple()])
            # find combinations for exact match groups
            exact = exact.groupby(id_exact)
            exact = exact.agg({col: list_notna for col in cols_exact})

            exact['df_index'] = exact['df_index'].apply(list_combo)
            exact = exact.explode('df_index')
            if self._index_mask['df2'] is not None:
                exact['df2_index'] = exact['df2_index'].apply(list_combo)
                exact = exact.explode('df2_index')


            exact['df_index','df_index_similar'] = pd.DataFrame(exact['df_index'].to_list(), columns=['df_index','df_index_similar'])


            if self._index_mask['df2'] is not None:
                exact = exact.apply(lambda x: list(product(x[0], x[1])), axis='columns')
                exact = exact.explode()
                exact = pd.DataFrame(exact.to_list(), columns=['df_index','df2_index'])
            score = pd.concat([score, exact])
            score['score'] = score['score'].fillna(1.0)


        # error check index inputs
        if index_df is None and index_df2 is None:
            raise RuntimeError('One of index_df or index_df2 must be provided.')
        elif index_df is not None and index_df2 is not None:
            raise RuntimeError('Only one of index_df or index_df2 can be provided.')
        elif index_df is not None:
            check = pd.Series(index_df)
            missing = ~check.isin(processed['df_index'])
            if any(missing):
                raise RuntimeError(f'index_df provided is not a valid index: {list(check[missing])}')
        elif index_df2 is not None:
            check = pd.Series(index_df2)
            missing = ~check.isin(processed['df2_index'])
            if any(missing):
                raise RuntimeError(f'index_df2 provided is not a valid index: {list(check[missing])}')

        # select indices by parameters given
        if index_df2 is None:
            comparison = processed.loc[processed['df_index'].isin(index_df), ['df_index', category]]
            comparison = comparison.merge(score, on='df_index')
            if self._index_mask['df2'] is not None:
                related = comparison[['df_index',category,'df2_index']].dropna(subset='df2_index')
                related = related.merge(score[['df2_index','df2_index_similar','score']], on='df2_index')
                related = related.dropna(subset='df2_index_similar')
                related = related.drop(columns=['df2_index'])
                comparison = pd.concat([comparison, related])
        else:
            comparison = processed.loc[processed['df2_index'].isin(index_df2), ['df2_index', category]]
            comparison = comparison.merge(score, on='df2_index')
            # include similar values within the same dataframe
            related = comparison[['df2_index',category,'df_index']].dropna(subset='df_index')
            related = related.merge(score[['df_index','df_index_similar','score']], on='df_index')
            related = related.dropna(subset='df_index_similar')
            related = related.drop(columns=['df_index'])
            comparison = pd.concat([comparison, related])
        

        # add similar values from first df
        similar = processed[['df_index', category]]
        similar = similar.dropna(subset='df_index')
        similar = similar.rename(columns={'df_index': 'df_index_similar'})
        comparison = comparison.merge(similar, on='df_index_similar', suffixes=('','_df_similar'), how='left')

        # add similar values from second df
        if self._index_mask['df2'] is not None:
            similar = processed[['df2_index', category]]
            similar = similar.dropna(subset='df2_index')
            similar = similar.rename(columns={'df2_index': 'df2_index_similar'})
            comparison = comparison.merge(similar, on='df2_index_similar', suffixes=('','_df2_similar'), how='left')

        # sort by first appearing index
        if index_df2 is not None:
            comparison = comparison.sort_values('df2_index')
        else:
            comparison = comparison.sort_values('df_index')

        # reset index that doesn't carry meaning for equality testing
        comparison = comparison.reset_index(drop=True)

        return comparison

    def __series_graph(self, edges):
        '''Convert Pandas series of lists to graph.'''

        edges = edges.apply(lambda x: list(combinations(x,2)))

        graph = nx.Graph()
        for e in edges.values:
            graph.add_edges_from(e)
        
        return graph


    def __assign_id(self, graph, id_name):

        df_id = pd.DataFrame({'index': list(nx.connected_components(graph))})
        df_id[id_name] = range(0, len(df_id))
        df_id = df_id.explode('index')

        return df_id


    def __common_features(self):

        df_feature = []
        for category,related in self.related.items():

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