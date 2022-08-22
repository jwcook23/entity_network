'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from collections import OrderedDict
import json

import pandas as pd

from entity_network import _index, _prepare, _compare_records, _network_helpers, _exceptions, _debug
from entity_network.clean_text import settings
from entity_network._performance_tracker import operation_tracker
from entity_network.network_plotter import network_dashboard

class entity_resolver(operation_tracker, network_dashboard):


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        # read global settings file
        self.settings = settings

        # assign globally unique node value
        self._df, self._index_mask = _index.assign_node(df, df2)

        # preprocessed text values
        self._compared_values = {}

        # outputs from compare method
        self.network_feature = {}
        self._similarity_score = {}
        self._compared_columns = OrderedDict([('name',None)])
        
        # outputs from network method
        self.network_id, self.network_map, self.entity_map = [None]*3

        # initialize performance time tracking and logging
        operation_tracker.__init__(self)

    def compare(self, category, columns, threshold:float=1, kneighbors:int=10):

        # input arguments
        if not category in settings.keys():
            raise _exceptions.InvalidCategory(f'Argument category must be one of {list(settings.keys())}')
        if not isinstance(kneighbors, int) or kneighbors<0:
            raise _exceptions.KneighborsRange('Argument kneighbors must be a positive integer.')
        if threshold<=0 or threshold>1:
            raise _exceptions.ThresholdRange('Argument threshold must be >0 and <=1.')

        # initialize timer for tracking duration
        self.reset_time()

        # create a single column, possibly composed of multiple columns for a category or split columns to be combined
        self._compared_values[category], self._compared_columns[category] = _prepare.flatten(self._df, columns, category)
        self.track('compare', '_prepare', 'flatten', category)

        # clean column text
        text_cleaner = self.settings[category]['cleaner']
        self._compared_values[category] = _prepare.clean(self._compared_values[category], category, text_cleaner)
        self.track('compare', '_prepare', 'clean', category)

        # find exact matches
        related_feature, self._df_exact = _compare_records.exact_match(self._compared_values[category])
        self.track('compare', '_compare_records', 'exact_match', category)

        # find similar matches
        id_category = f'{category}_id'
        if threshold==1:
            # skip finding similar matches due to increased processed requirements and non-exact fuzzy matching
            related_feature['id_similar'] = pd.NA
            related_feature[id_category] = related_feature['id_exact']
            similar_score = None
            self.track('compare', None, None, 'skip similar')
        else:

            # create term frequency–inverse document frequency matrix to numerically compare text
            text_comparer = self.settings[category]['comparer']
            tfidf, tfidf_index = _compare_records.create_tfidf(self._compared_values[category], text_comparer)
            self.track('compare', '_compare_records', 'create_tfidf', category)

            # find similar text values using a non-blocking k-nearest neighbor approach
            similar_score = _compare_records.similar_match(tfidf, tfidf_index, kneighbors)
            self.track('compare', '_compare_records', 'similar_match', category)

            # assign an overall id to similar records using connected components
            similar_feature = _compare_records.similar_id(similar_score, tfidf_index, threshold)
            self.track('compare', '_compare_records', 'similar_id', category)

            # expand similarity score after an id was assigned using connected components
            similar_score = _compare_records.expand_score(similar_score, similar_feature, threshold)
            self.track('compare', '_compare_records', 'expand_score', category)

            # determine an overall id using connected components of similar and exact matches
            related_feature, similar_feature = _compare_records.combined_id(related_feature, similar_feature, id_category)
            self.track('compare', '_compare_records', 'combined_id', category)

            # include duplicated values in the first df related to a value in the second
            related_feature, similar_score = _compare_records.fill_exact(related_feature, similar_score, self._df_exact)
            self.track('compare', '_compare_records', 'fill_exact', category)

        # remove matches that do not match another index (columns for a category may contain the same value for a given record)
        related_feature, similar_score = _compare_records.remove_self(related_feature, similar_score, id_category)
        self.track('compare', '_compare_records', 'remove_self', category)

        # assign the original index
        related_feature, similar_score = _compare_records.translate_index(related_feature, similar_score, self._index_mask, id_category)
        self.track('compare', '_compare_records', 'translate_index', category)

        # store similarity for debugging
        self._similarity_score[category] = similar_score

        # store features for forming network and entity resolution
        self.network_feature[category] = related_feature

        return related_feature, similar_score


    def network(self):

        # initialize timer for tracking duration
        self.reset_time()

        # form matrix of indices connected on any feature
        self.network_map = _network_helpers.combine_features(self.network_feature)
        self.track('network', '_network_helpers', 'combine_features', None)

        # determine an overall id using indices connected on any feature
        self.network_id, self.network_map = _network_helpers.assign_id(self.network_map)
        self.track('network', '_network_helpers', 'assign_id', None)

        # assign the original index
        self.network_id, self.network_map = _network_helpers.translate_index(self.network_id, self.network_map, self._index_mask)
        self.track('network', '_network_helpers', 'translate_index', None)

        # resolve entities if names were compared
        if 'name' in self.network_feature:
            self.entity_map, self.network_map = _network_helpers.resolve_entity(self.network_map, self.network_feature, self._df['df'])
            self.track('network', '_network_helpers', 'resolve_entity', None)

        # summerize the network by connections or by entity if names were compared
        if self.entity_map is None:
            self.network_summary = _network_helpers.summerize_connections(self.network_id, self.network_feature, self._compared_values)
            self.track('network', '_network_helpers', 'summerize_connections', None)
        else:
            self.network_summary = _network_helpers.summerize_entity(self.network_map, self._compared_columns, self._df['df'])
            self.track('network', '_network_helpers', 'summerize_entity', None)
            

    def debug_similar(self, category, cluster_edge_limit=5):

        # select similarity score for given category
        score = self._similarity_score[category]
        
        # set node and node_similar as columns for merging in processed values
        score = score.reset_index()

        # add the main processed/cleaned values from the first df
        score = _debug.first_df(score, self._compared_values[category], category, self._df_exact)

        # add the similar processed/cleaned values from the first or second df
        score = _debug.similar_values(score, self._compared_values[category], category)     

        # remove columns used to track record and dataframe
        score = score.drop(columns=['node','node_similar'])

        # group similar values by id in order of decreasing similarity score
        score = score.sort_values(by=['id_similar','score'], ascending=[True, False])

        # find values nearest the cluster edge
        in_cluster, out_cluster = _debug.cluster_edge(score, cluster_edge_limit)
        
        # calculate term difference between processed values for items nearest cluster edge
        in_cluster, out_cluster = _debug.record_difference(score, in_cluster, out_cluster, category)

        return score, in_cluster, out_cluster

    def plot_network(self):

        network_dashboard.__init__(self)