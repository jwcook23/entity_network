from collections import OrderedDict
import json

import pandas as pd

from entity_network import _index, _prepare, _compare_records, _network_helpers, _exceptions, _debug
from entity_network.clean_text import comparison_rules
from entity_network._performance_tracker import operation_tracker
from entity_network.network_plotter import network_dashboard

class entity_resolver(operation_tracker, network_dashboard):


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):
        ''' Find links in a single dataframe or two dataframes for
        entity resolution and/or network link analysis.

        TODO: validate documentation

        Parameters
        ----------
        df (pandas.DataFrame): first dataframe containing entity features
        df2 (pandas.DataFrame, default=None): second dataframe containing entity features

        Properties
        ----------
        er.network_id (pd.DataFrame): 
        er.network_map (pd.DataFrame):
        er.entity_map (pd.DataFrame | None): 
        er.network_summary (pd.DataFrame): 

        Examples
        --------

        Resolve entities and find networks in a single dataframe.

        >>> df = pd.DataFrame(columns=['Name','Phone','Address'])
        >>> er = entity_resolver(df)
        >>> er.compare('name', columns='Name')
        >>> er.compare('phone', columns='Phone')
        >>> er.compare('address', columns='Address', threshold=0.9)
        >>> er.network()

        Resolve entities and find networks in two dataframes.

        >>> df = pd.DataFrame(columns=['NameCol1','PhoneCol1','AddressCol1'])
        >>> df2 = pd.DataFrame(columns=['NameCol2','PhoneNameCol2','AddressCol2'])
        >>> er = entity_resolver(df, df2)
        >>> er.compare('name', columns={'df': 'NameCol1', 'df2': 'NameCol2'})
        >>> er.compare('phone', columns={'df': 'Phone1', 'df2': 'PhoneCol2'})
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': 'AddressCol2'}, threshold=0.9)
        >>> er.network()

        Find networks in two dataframes. Entities aren't resolved as name are not compared.

        >>> df = pd.DataFrame(columns=['NameCol1','PhoneCol1','AddressCol1'])
        >>> df2 = pd.DataFrame(columns=['NameCol2','PhoneNameCol2','AddressCol2'])
        >>> er = entity_resolver(df, df2)
        >>> er.compare('phone', columns={'df': 'Phone1', 'df2': 'PhoneCol2'})
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': 'AddressCol2'}, threshold=0.9)
        >>> er.network()

        See Also
        --------
        compare: methods to compare values
        network: resolve entities and form final network relationships

        '''

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
        ''' Compare columns in a single dataframe or two dataframes to find relationships
        used to resolve entities and find networks.

        Parameters
        ----------
        category (str): 
        columns (str|list|dict): 
        thresold (float, default=1): 
        kneighbors (int): TODO: test and document affect of kneighbors

        Examples
        --------

        Compare values in a single dataframe.

        >>> er = entity_resolver(df)
        >>> er.compare('name', columns='Name')

        Compare values between two dataframes with a similarity threshold.

        >>> er = entity_resolver(df1, df2)
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': 'AddressCol2'}, threshold=0.9)

        Compare values that are spread across multiple columns.

        >>> er = entity_resolver(df, df2)
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': ['Line1','City','State','Zip']}, threshold=0.9)

        See Also
        --------
        network: resolve entities and form final network relationships

        '''

        # input arguments
        if not category in comparison_rules.keys():
            raise _exceptions.InvalidCategory(f'Argument category must be one of {list(comparison_rules.keys())}')
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
        text_cleaner = comparison_rules[category]['cleaner']
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
            text_comparer = comparison_rules[category]['comparer']
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
        ''' Summerize network relationships and resolve entities if names were compared.

        Parameters
        ----------
        None

        Examples
        --------
        Resolve entities and find networks in two dataframes.

        >>> df = pd.DataFrame(columns=['NameCol1','PhoneCol1','AddressCol1'])
        >>> df2 = pd.DataFrame(columns=['NameCol2','PhoneNameCol2','AddressCol2'])
        >>> er = entity_resolver(df, df2)
        >>> er.compare('name', columns={'df': 'NameCol1', 'df2': 'NameCol2'})
        >>> er.compare('phone', columns={'df': 'Phone1', 'df2': 'PhoneCol2'})
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': 'AddressCol2'}, threshold=0.9)
        >>> er.network()

        Find networks in two dataframes. Entities aren't resolved as name are not compared.

        >>> df = pd.DataFrame(columns=['NameCol1','PhoneCol1','AddressCol1'])
        >>> df2 = pd.DataFrame(columns=['NameCol2','PhoneNameCol2','AddressCol2'])
        >>> er = entity_resolver(df, df2)
        >>> er.compare('phone', columns={'df': 'Phone1', 'df2': 'PhoneCol2'})
        >>> er.compare('address', columns={'df': 'AddressCol1', 'df2': 'AddressCol2'}, threshold=0.9)
        >>> er.network()   

        See Also
        --------
        compare: methods to compare values    
        '''

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