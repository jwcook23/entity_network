'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from unicodedata import category
import pandas as pd
import networkx as nx

from entity_network import _exceptions, _core

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._prepare_index(df, df2)          

        self.network_relation = {}
        self.similar_records = {}
        self.network_feature = None
        self.network_id = None
        # self.entity_feature = None
        # self.entity_id = None
        self._processed = {}


    def compare(self, category, columns, kneighbors:int=10, threshold:int=1, analyzer='default', preprocessor='default'):

        # prepare values for comparison
        self._processed[category] = _core.prepare_values(self._df, category, columns, preprocessor)
        self._processed[category].index.names = ('index', 'column')
        self._processed[category].name = category

        # ignore values the processor completely removed
        # self._processed[category] = self._processed[category].dropna()

        # compare values on similarity threshold
        related, similar = _core.find_related(category, self._processed[category], kneighbors=kneighbors, threshold=threshold, analyzer=analyzer)
        self.network_relation[category] = related
        self.similar_records[category] = similar


    def entity(self, columns, kneighbors, threshold, analyzer='default'):

        raise NotImplemented('Only compare is currently implemented.')

        # self.compare('name', columns, kneighbors, threshold, analyzer)

        # # form graph of matching names
        # name = _conversion.from_pandas_df_id(self.network_relation['name'], 'name_id')

        # # compare name graph to graphs of other features to resolve entities
        # self.entity_feature = []
        # for c in self._other:
        #     other = _conversion.from_pandas_df_id(self.network_relation[c], f'{c}_id')
        #     other = nx.intersection(name, other)
        #     other.remove_nodes_from(list(nx.isolates(other)))
        #     other = list(nx.connected_components(other))
        #     other = pd.DataFrame({'index': other})
        #     other['category'] = c
        #     self.entity_feature.append(other)
        # self.entity_feature = pd.concat(self.entity_feature, ignore_index=True)

        # # assign entity_id to all records
        # self.entity_id, self.entity_feature = self._core.assign_id(self.entity_feature, 'entity_id')

        # # assign entity_name
        # self.entity_id = self._assign_name(self.entity_id, 'entity_id', 'entity_name')

        # # assign entity_id to original dataframe
        # self._df = self._df.merge(self.entity_id[['entity_id']], left_index=True, right_index=True, how='left')
        # self._df['entity_id'] = self._df['entity_id'].astype('Int64')

        # return self._df, self.entity_id, self.entity_feature


    def network(self):

        # determine network by matching features
        self.network_feature = []
        for category,related in self.network_relation.items():
            other = related.reset_index()
            other = other.groupby(f'{category}_id')
            other = other.agg({'index': set})
            other['category'] = category
            self.network_feature.append(other)
        self.network_feature = pd.concat(self.network_feature, ignore_index=True)

        # assign network_id to connected records
        self.network_id, self.network_feature = _core.assign_id(self.network_feature, 'network_id')

        # translate index back to the original values
        self._original_index()

        return self.network_id, self.network_feature, self.network_relation

    def _prepare_index(self, df, df2):

        if df.index.has_duplicates:
            raise _exceptions.DuplicatedIndex('Argument df index must be unique.')
        if df2 is not None and df2.index.has_duplicates:
            raise _exceptions.DuplicatedIndex('Argument df2 index must be unique.')

        # develop unique integer based index for each df in case they need to be combined
        index_mask = {
            'df': pd.Series(df.index, index=range(0, len(df)))
        }
        index_mask['df'].name = 'index'
        df.index = index_mask['df'].index
        if df2 is not None:
            # start df2 index at end of df index
            seed = len(index_mask['df'])+1
            index_mask['df2'] = pd.Series(df2.index, index=range(seed, seed+len(df2)))
            index_mask['df2'].name = 'index'
            df2.index = index_mask['df2'].index
            # stack df2 on each of df for a single df to be compared
            df = pd.concat([df, df2])

        self._df = df
        self._index_mask = index_mask

    def _original_index(self):

        # TODO: split self.similar_records
        for category, combined in self.network_relation.items():
            self.network_relation[category] = self._split_df(combined)
        self.network_feature = self._split_df(self.network_feature)
        self.network_id = self._split_df(self.network_id)


    def _split_df(self, combined):
        '''Split dataframe into original dataframes and add the original index.'''
        df = combined.merge(
            self._index_mask['df'], left_index=True, right_index=True
        )
        df = df.set_index('index')
        df2 = combined.merge(
            self._index_mask['df2'], left_index=True, right_index=True
        )
        df2 = df2.set_index('index')

        return {'df': df, 'df2': df2}