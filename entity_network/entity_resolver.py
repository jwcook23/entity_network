'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations

import pandas as pd
import networkx as nx

from entity_network import _index, _prepare, _compare

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _index.unique(df, df2)

        self.relationship = {}
        # self.similar_records = {}
        # self.network_feature = None
        # self.network_id = None
        # self.entity_feature = None
        # self.entity_id = None
        self.processed = {}
        self.timer = pd.DataFrame(columns=['caller','file','method','category','time_seconds'])


    def compare(self, category, columns, kneighbors:int=10, threshold:int=1, text_comparer='default', text_cleaner='default'):

        # combine split columns, flatten into single
        print(f'flattening {columns}')
        tstart = time()
        self.processed[category] = _prepare.flatten(self._df, columns)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'flatten', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # clean column text
        print(f'cleaning {category}')
        tstart = time()
        self.processed[category] = _prepare.clean(self.processed[category], category, text_cleaner)
        self.processed[category].index.names = ('index', 'column')
        self.processed[category].name = category
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'clean', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # ignore values the processor completely removed
        # self.processed[category] = self.processed[category].dropna()

        # compare values on similarity threshold
        print(f'comparing {columns}')
        tstart = time()
        related, similar = _compare.match(category, self.processed[category], kneighbors=kneighbors, threshold=threshold, text_comparer=text_comparer)
        self.relationship[category] = related
        # self.similar_records[category] = similar
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

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
        network_id, network_feature = _index.original(self._index_mask, network_id, network_feature)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_index', 'original', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return network_id, network_feature


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
        for category,related in self.relationship.items():

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