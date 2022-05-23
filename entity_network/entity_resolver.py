'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from unicodedata import category
import pandas as pd
import networkx as nx

from entity_network import _exceptions, _core, _conversion

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _core.prepare_index(df, df2)

        self.relationship = {}
        # self.similar_records = {}
        # self.network_feature = None
        # self.network_id = None
        # self.entity_feature = None
        # self.entity_id = None
        self.processed = {}


    def compare(self, category, columns, kneighbors:int=10, threshold:int=1, analyzer='default', preprocessor='default'):

        # prepare values for comparison
        self.processed[category] = _core.prepare_values(self._df, category, columns, preprocessor)
        self.processed[category].index.names = ('index', 'column')
        self.processed[category].name = category

        # ignore values the processor completely removed
        # self.processed[category] = self.processed[category].dropna()

        # compare values on similarity threshold
        related, similar = _core.find_related(category, self.processed[category], kneighbors=kneighbors, threshold=threshold, analyzer=analyzer)
        self.relationship[category] = related
        # self.similar_records[category] = similar


    def entity(self, columns, kneighbors, threshold, analyzer='default'):

        raise NotImplemented('Only compare is currently implemented.')

        # self.compare('name', columns, kneighbors, threshold, analyzer)

        # # form graph of matching names
        # name = _conversion.from_pandas_df_id(self.relationship['name'], 'name_id')

        # # compare name graph to graphs of other features to resolve entities
        # self.entity_feature = []
        # for c in self._other:
        #     other = _conversion.from_pandas_df_id(self.relationship[c], f'{c}_id')
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

    def network(self, additional_details: list = []):

        # determine network by matching features
        network_feature = []
        for category,related in self.relationship.items():
            feature = related.reset_index()
            feature = feature.groupby(f'{category}_id')
            feature = feature.agg({'index': tuple, 'column': tuple})
            network_feature.append(feature)
        network_feature = pd.concat(network_feature, ignore_index=True)
        network_feature.index.name = 'feature_id'

        # create network graph
        self.network_graph = _conversion.from_pandas_series(network_feature['index'])

        # assign network_id to connected records
        network_id = pd.DataFrame({'index': list(nx.connected_components(self.network_graph))})
        network_id['network_id'] = range(0, len(network_id))
        network_id = network_id.explode('index')

        # add original index for df
        network_id = network_id.merge(self._index_mask['df'], left_on='index', right_index=True, how='left')
        network_id['df_index'] = network_id['df_index'].astype('Int64')

        # add original index for df
        if self._index_mask['df2'] is not None:
            network_id = network_id.merge(self._index_mask['df2'], left_on='index', right_index=True, how='left')
            network_id['df2_index'] = network_id['df2_index'].astype('Int64')

        # add original index to network_feature
        network_feature = network_feature.apply(pd.Series.explode)
        network_feature = network_feature.reset_index()
        network_feature = network_feature.merge(network_id[network_id.columns.drop('network_id')], on='index')

        # remove artificially created index column
        # network_id = network_id.drop(columns=['index'])
        # network_feature = network_feature.drop(columns=['index'])
        network_id = network_id.set_index('index')
        network_id.index.name = None
        network_feature = network_feature.set_index('index')
        network_feature.index.name = None

        # add node details
        for index in self.network_graph.nodes:
            feature = self._df.loc[index, network_feature.loc[[index], 'column']].to_dict()
            self.network_graph.nodes[index].update(feature)
        for col in additional_details:
            for index in self.network_graph.nodes:
                name = {col: self._df.at[index, col]}
                self.network_graph.nodes[index].update(name)

        return network_id, network_feature
