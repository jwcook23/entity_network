'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from collections import OrderedDict

import pandas as pd

from entity_network import _index, _prepare, _compare, _helpers, _exceptions
from entity_network.network_plotter import network_dashboard

class entity_resolver(network_dashboard):


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _index.unique(df, df2)

        # preprocessed text values
        self.processed = {}

        # outputs from compare method
        self.network_feature = {}
        self.similar_score = {}
        self.compared_columns = OrderedDict([('name',None)])
        
        # outputs from network method
        self.network_feature = {}
        self.network_id, self.network_map = [None]*2

        # outputs form entity method
        # self.entity_feature, self.entity_id  = [None]*2
        
        self.timer = pd.DataFrame(columns=['caller','file','method','category','time_seconds'])


    def compare(self, category, columns, threshold:float=1, kneighbors:int=10, text_comparer='default', text_cleaner='default'):

        # input arguments
        if not isinstance(kneighbors, int) or kneighbors<0:
            raise _exceptions.KneighborsRange(f'Argument kneighbors must be a positive integer.')
        if threshold<=0 or threshold>1:
            raise _exceptions.ThresholdRange(f'Argument threshold must be >0 and <=1.')

        # combine split columns, flatten into single
        print(f'Combining columns then flattening into single column: {columns}.')
        tstart = time()
        self.processed[category], compared = _prepare.flatten(self._df, columns)
        self.compared_columns[category] = compared
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'flatten', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # clean column text
        if text_cleaner is not None:
            print(f'Cleaning category for comparison: {category}.')
            tstart = time()
            for frame, values in self.processed[category].items():
                if values is not None:
                    self.processed[category][frame] = _prepare.clean(values, category, text_cleaner)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'clean', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # set sources for tracability when comparing multiple categories
        for frame, values in self.processed[category].items():
            if values is not None:
                self.processed[category][frame].index.names = ('node', 'column')
                self.processed[category][frame].name = category

        # find exact matches
        print(f'Finding exact matches for category: {category}.')
        tstart = time()
        related_feature = _compare.exact_match(self.processed[category])
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'exact_match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # find similar matches
        id_category = f'{category}_id'
        if threshold==1:
            related_feature['id_similar'] = pd.NA
            related_feature[id_category] = related_feature['id_exact']
            similar_score = None
        else:
            print(f'Creating tfidf for category: {category}.')
            tstart = time()
            tfidf, tfidf_index = _compare.create_tfidf(category, self.processed[category], text_comparer, related_feature)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'create_tfidf', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

            print(f'Finding similar matches for category: {category}.')
            tstart = time()
            similar_score = _compare.similar_match(tfidf, tfidf_index, kneighbors)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'similar_match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

            print(f'Assigning a similar ID for category: {category}.')
            tstart = time()
            similar_feature = _compare.similar_id(similar_score, tfidf_index, threshold)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'similar_id', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

            print(f'Preparing similarity score for category: {category}.')
            tstart = time()
            similar_score = _compare.expand_score(similar_score, similar_feature, threshold)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'expand_score', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

            print(f'Determining overall ID using exact and similar for: {category}.')
            tstart = time()
            related_feature, similar_feature = _compare.combined_id(related_feature, similar_feature, id_category)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'combined_id', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # remove matches that do not match another index (can occur since multiple columns are flattened)
        print(f'Removing rows that only self-match for category: {category}.')
        tstart = time()
        related_feature, similar_score = _compare.remove_self_matches(related_feature, similar_score, id_category)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'remove_self_matches', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # assign the original index
        print(f'Assigning the original input index for category: {category}.')
        tstart = time()
        related_feature, similar_score = _compare.translate_index(related_feature, similar_score, self._index_mask, id_category)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'translate_index', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # store similarity for debugging
        self.similar_score[category] = similar_score

        # store features for forming network and entity resolution
        self.network_feature[category] = related_feature

        # add original index to processed values
        self.processed[category]['df'] = self.processed[category]['df'].reset_index()
        self.processed[category]['df'] = self.processed[category]['df'].merge(self._index_mask['df'], on='node')
        if self._index_mask['df2']is not None:
            self.processed[category]['df2'] = self.processed[category]['df2'].reset_index()
            self.processed[category]['df2'] = self.processed[category]['df2'].merge(self._index_mask['df2'], on='node')

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return related_feature, similar_score


    def network(self):

        # form matrix of indices connected on any feature
        print('Combining matching features into a single matrix.')
        tstart = time()
        network_map = _helpers.combine_features(self.network_feature)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_helpers', 'combine_features', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # determine an overall id using indices connected on any feature
        print('Assigning overall network ID.')
        tstart = time()
        network_id, network_map = _helpers.overall_id(network_map)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_helpers', 'overall_id', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # add original index
        tstart = time()
        self.network_id, self.network_map = _index.network(network_id, network_map, self._index_mask)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_index', 'network', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # resolve entity if needed
        if 'name' in self.network_feature:
            self._entity()

        # create summary of the network
        self._summary()

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return self.network_id, self.network_map, self.network_feature


    def _entity(self):

        # assume similar names in the same network are the same entity
        # TODO: require a single matching feature instead of the entire network?
        entity_id = self.network_map[['network_id']].merge(
            self.network_feature['name'][['name_id']], 
            left_index=True, right_index=True, how='left'
        )
        entity_id = entity_id.groupby(['network_id','name_id']).ngroup()

        # assume entity_id without a name_id
        assume = entity_id==-1
        seed = entity_id.max()+1
        if pd.isna(seed):
            seed = 0
        entity_id[assume] = range(seed, seed+sum(assume))

        # assign resolved entity to network map
        self.network_map['entity_id'] = entity_id

        # determine unique features belonging to each entity
        self.entity = pd.DataFrame(columns=['network_id','entity_id','category','column','value'])
        for category, feature in self.network_feature.items():
            category_id = f'{category}_id'
            # determine matching column
            details = self.network_map[['network_id','entity_id']]
            details = details.merge(feature[['column', category_id]], on='node', how='left')
            # add value from matching column
            # TODO: handle possibly two dataframes
            columns = details['column'].dropna().unique()
            value = self._df['df'][columns].stack()
            value.name = 'value'
            details = details.merge(value, left_on=['node','column'], right_index=True)
            # remove duplicated info
            details = details.drop_duplicates()
            # add source category
            details['category'] = category
            # combine details from each features
            self.entity = pd.concat([self.entity, details])
            self.entity[category_id] = self.entity[category_id].astype('Int64')

    def _summary(self):

        # TODO: include other variations of summerizing a network, including for two dataframes
        # TODO: handle summary for network without an entity id
        network_summary = self.network_map.groupby('network_id')
        network_summary = network_summary.agg({'entity_id': 'nunique'})
        network_summary = network_summary.rename(columns={'entity_id': 'entity_count'})
        network_summary = network_summary.sort_values('entity_count', ascending=False)

        self.network_summary = network_summary

    def debug_similar(self, category, cluster_edge_limit=5):
        
        comparer = {'address': _compare.address}

        # extra debugging info for given category
        similar = self.similar_score[category]

        # add processed values into similar score for the first dataframe
        df_categories = {'exact': f'df_{category}', 'similar': f'df_{category}_similar'}
        processed = self.processed[category]['df']
        processed = processed[['df_index', category]].dropna(subset='df_index')
        similar = similar.merge(
            processed.rename(columns={category: df_categories['exact']}), 
            on='df_index', how='left'
        )
        similar = similar.merge(
            processed.rename(columns={category: df_categories['similar'], 'df_index': 'df_index_similar'}),
            on='df_index_similar', how='left',
        )
        # add processed values into similar score for the second dataframe 
        if self._index_mask['df2'] is None:
            df2_categories = {'exact': None, 'similar': None}
        else:
            df2_categories = {'exact': f'df2_{category}', 'similar': f'df2_{category}_similar'}
            processed = self.processed[category][['df2_index',category]].dropna(subset='df2_index')
            similar = similar.merge(
                processed[['df2_index',category]].dropna().rename(columns={category: df2_categories['exact']}), 
                on='df2_index', how='left'
            )
            similar = similar.merge(
                processed.dropna().rename(columns={category: df2_categories['similar'], 'df2_index': 'df2_index_similar'}),
                on='df2_index_similar', how='left',
            )

        # group by id and score
        similar = similar.sort_values(by=['id_similar','score'], ascending=[True, False])

        # split by in/out of cluster with closest distance to cluster edge appearing first and return comparison of difference
        # TODO: provide a summary of differences
        # TODO: provide elbow diagram of score to help determine where the threshold should be (change point)
        columns = list(df_categories.values())+list(df2_categories.values())
        columns = [x for x in columns if x is not None]
        in_cluster = similar[similar['threshold']].sort_values(by='score', ascending=True).head(cluster_edge_limit)
        in_cluster[f'{category}_difference'] = in_cluster[columns].apply(comparer[category], axis=1)
        out_cluster = similar[~similar['threshold']].sort_values(by='score', ascending=False).head(cluster_edge_limit)
        out_cluster[f'{category}_difference'] = out_cluster[columns].apply(comparer[category], axis=1)

        return similar, in_cluster, out_cluster

    def plot_network(self):
        
        # network_ploter.server(self.network_map, self.network_feature, self.entity)

        network_dashboard.__init__(self)