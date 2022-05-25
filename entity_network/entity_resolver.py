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


    def network(self):

        # determine network by matching features
        network_feature = _core.common_features(self.relationship)

        # create network graph
        self.network_graph = _conversion.from_pandas_series(network_feature['index'])

        # assign network_id to connected records
        network_id = _core.assign_id(self.network_graph, 'network_id')

        # add original index
        network_id, network_feature = _core.original_id_index(self._index_mask, network_id, network_feature)

        return network_id, network_feature
