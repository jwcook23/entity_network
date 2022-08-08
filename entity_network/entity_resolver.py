'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from collections import OrderedDict

import pandas as pd

from entity_network import _index, _prepare, _compare, _helpers, _exceptions
from entity_network._difference import term_difference
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
        self.network_id, self.network_map, self.entity_map = [None]*3
        
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
        related_feature, self._df_exact = _compare.exact_match(self.processed[category])
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

            # include duplicated values in the first df related to a value in the second
            if self._df_exact is not None:
                fill = related_feature.set_index('node').drop(columns='column')
                fill = self._df_exact.merge(fill, left_index=True, right_index=True)
                fill = fill.drop(columns='id')
                related_feature = pd.concat([related_feature, fill], ignore_index=True)

                fill = similar_score.reset_index().set_index('node').drop(columns='column')
                fill = self._df_exact.merge(fill, left_index=True, right_index=True)
                fill = fill.set_index('node_similar')
                fill = fill.drop(columns='id')
                similar_score = pd.concat([similar_score, fill], ignore_index=False)

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
        # self.processed[category]['df'] = self.processed[category]['df'].reset_index()
        # self.processed[category]['df'] = self.processed[category]['df'].merge(self._index_mask['df'], on='node')
        # if self._index_mask['df2']is not None:
        #     self.processed[category]['df2'] = self.processed[category]['df2'].reset_index()
        #     self.processed[category]['df2'] = self.processed[category]['df2'].merge(self._index_mask['df2'], on='node')

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

        # resolve entity if needed names were also compared
        if 'name' in self.network_feature:
            self._entity()

        # summerize the network
        if self.entity_map is not None:
            self.network_summary = self._summerize_entity()
        else:
            self.network_summary = self._summerize_dfs()

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)


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
        self.entity_map = pd.DataFrame(columns=['network_id','entity_id','category','column','value'])
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
            self.entity_map = pd.concat([self.entity_map, details])
            self.entity_map[category_id] = self.entity_map[category_id].astype('Int64')

    def _summerize_entity(self):

        network_summary = self.network_map.groupby('network_id')
        network_summary = network_summary.agg({'entity_id': 'nunique'})
        network_summary = network_summary.rename(columns={'entity_id': 'entity_count'})
        network_summary = network_summary.sort_values('entity_count', ascending=False)

        return network_summary

    def _summerize_dfs(self):

        if 'df2_index' not in self.network_id:
            network_summary = None
            return network_summary

        network_summary = self.network_id[['network_id','df_index']].dropna().groupby('network_id').agg({'df_index': list})
        network_summary = network_summary.merge(
            self.network_id[['network_id','df2_index']].dropna().groupby('network_id').agg({'df2_index': list}),
            left_index=True, right_index=True
        )
        network_summary = network_summary.explode('df2_index')
        network_summary = network_summary.explode('df_index')
        feature = pd.DataFrame()
        for category, data in self.network_feature.items():
            source = data[['df_index']].dropna()
            if len(source)>0:
                source['df_feature'] = category
                feature = pd.concat([feature, source], axis=0)
            source = data[['df2_index']].dropna()
            if len(source)>0:
                source['df2_feature'] = category
            feature = pd.concat([feature, source], axis=0)
        network_summary = network_summary.merge(feature.groupby('df_index').agg({'df_feature': list}), on='df_index', how='left')
        network_summary['df_feature'] = network_summary['df_feature'].apply(lambda x: ','.join(x))
        network_summary = network_summary.merge(feature.groupby('df2_index').agg({'df2_feature': list}), on='df2_index', how='left')
        network_summary['df2_feature'] = network_summary['df2_feature'].apply(lambda x: ','.join(x))

        return network_summary

    def debug_similar(self, category, cluster_edge_limit=5):

        # select similarity score for given category
        score = self.similar_score[category]
        
        # set node and node_similar as columns for merging in processed values
        score = score.reset_index()

        # add the main processed/cleaned values from the first df
        processed = self.processed[category]['df'].copy()
        processed.name = f'{category}_df_value'
        # include exact matches within the first df
        if self._df_exact is not None:
            fill = processed.reset_index(level=1, drop=True)
            fill = self._df_exact.merge(fill, left_index=True, right_index=True)
            fill = fill.drop(columns='id')
            fill = fill.set_index(keys=['node','column'])
            fill = fill[processed.name]
            processed = pd.concat([processed, fill], ignore_index=False)
        score = score.merge(processed, how='left', on=['node','column'])

        # add the similar processed/cleaned values from the first or second df
        if self.processed[category]['df2'] is None:
            processed = self.processed[category]['df'].copy()
            processed.name = f'{category}_df_similar_value'
            score = score.merge(processed, how='left', left_on=['node_similar','column'], right_on=['node','column'])
        else:
            processed = self.processed[category]['df2'].copy().reset_index()
            processed = processed.rename(columns={'node': 'node_similar', category: f'{category}_df2_similar_value', 'column': 'column_df2'})
            score = score.merge(processed, how='left', on='node_similar')            

        # remove columns used to track record and dataframe
        # score = score.set_index(keys=['node','node_similar'])
        score = score.drop(columns=['node','node_similar'])

        # group similar values by id in order of decreasing similarity score
        score = score.sort_values(by=['id_similar','score'], ascending=[True, False])

        # find values closest to not being included
        in_cluster = score[score['threshold']].sort_values(by='score', ascending=True)
        in_cluster = in_cluster.head(cluster_edge_limit)
        
        # find records closest to belonging to a cluster
        out_cluster = score[~score['threshold']].sort_values(by='score', ascending=False)
        out_cluster = out_cluster.head(cluster_edge_limit)
        
        # calculate term difference between processed values for items nearest cluster edge
        compare = score.columns[score.columns.str.endswith('_value')]
        diff = compare.str.replace('_value$', '_diff', regex=True)
        in_cluster[[diff[0],diff[1]]] = in_cluster[compare].apply(term_difference[category], axis=1, result_type='expand')
        out_cluster[[diff[0],diff[1]]] = out_cluster[compare].apply(term_difference[category], axis=1, result_type='expand')


        return score, in_cluster, out_cluster

    def plot_network(self):

        network_dashboard.__init__(self)