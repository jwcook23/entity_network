'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations, chain
from collections import OrderedDict

import pandas as pd
import networkx as nx
from bokeh.models import Circle, MultiLine, HoverTool
from bokeh.plotting import figure, from_networkx, show, output_file

from entity_network import _index, _prepare, _compare, _helpers, _exceptions

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _index.unique(df, df2)

        # preprocessed text values
        self.processed = {}

        # outputs from compare method
        self.network_feature = {}
        self.similar_score = {}
        
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
        self.processed[category] = _prepare.flatten(self._df, columns)
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

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return self.network_id, self.network_map, self.network_feature

    def _entity(self):

        # assume similar names in the same network are the same entity
        # TODO: require a single matching feature instead of the entire network?
        entity = self.network_map[['network_id']].merge(
            self.network_feature['name'][['name_id']], 
            left_index=True, right_index=True, how='left'
        )
        entity = entity.groupby(['network_id','name_id']).ngroup()

        # assume entity_id without a name_id
        assume = entity==-1
        seed = entity.max()+1
        if pd.isna(seed):
            seed = 0
        entity[assume] = range(seed, seed+sum(assume))

        self.network_map['entity_id'] = entity
    

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

    def plot_network(self, file_name):
    # http://docs.bokeh.org/en/latest/docs/gallery/network_graph.html
    # https://docs.bokeh.org/en/latest/docs/user_guide/graph.html

        G = nx.Graph()

        # for category, feature in self.network_feature:
        # category = 'address'
        # feature = self.network_feature[category]
        # feature = feature.copy().reset_index()

        network = self.network_map

        # calculate network summary
        # TODO: include in base class
        network_summary = network.groupby('network_id')
        network_summary = network_summary.agg({'entity_id': 'nunique'})
        network_summary = network_summary.rename(columns={'entity_id': 'entity_count'})
        network_summary = network_summary.sort_values('entity_count', ascending=False)

        network = network[network['network_id']==network_summary.index[0]]

        # remove duplicates for an entity
        network = network.drop_duplicates(subset=['entity_id','address_id','phone_id','email_id'])

        # track unique column sources for hover display ensuring name if first
        source = list(self.network_feature.keys())
        source.remove('name')
        source = ['name']+source
        source = OrderedDict(zip(source, [None]*len(source)))

        # aggregate features for each entity
        entity = pd.DataFrame(columns=['entity_id','column','value'])
        for category in source.keys():
            # determine matching column
            details = network[['entity_id']]
            details = details.merge(self.network_feature[category][['column']], on='node', how='left')
            # add value from matching column
            columns = details['column'].dropna().unique()
            source[category] = columns
            value = self._df['df'][columns].stack()
            value.name = 'value'
            details = details.merge(value, left_on=['node','column'], right_index=True)
            # remove duplicated info
            details = details.drop_duplicates()
            # combine details from each features
            entity = pd.concat([entity, details])
        # aggreate single unique values for each source column
        entity = entity.groupby(['entity_id', 'column'])
        entity = entity.agg({'value': 'unique'})
        entity['value'] = entity['value'].apply(lambda x: '<br>'.join(x))
        entity = entity.reset_index()
        # aggreate nodes for networkx node attributes
        def attrs(df):
            values = dict((zip(df['column'],df['value'])))
            values['Entity ID'] = df.iloc[0,0]
            return values
        entity = entity.groupby('entity_id')
        entity = entity.apply(attrs)
        # 
        entity = list(zip(entity.index, entity.values))
        G.add_nodes_from(entity)


        # add edges
        for category in self.network_feature.keys():
            if category=='name':
                continue
            edges = network.groupby(f'{category}_id')
            edges = edges.agg({
                'entity_id': list
            })
            if len(edges)==0:
                continue
            edges = edges[edges['entity_id'].str.len()>1]
            # TODO: add 3-tuple where the 3rd is the edge attribute describing how the connection is made
            edges['entity_id'] = edges['entity_id'].apply(lambda x: list(zip(x[0:-1], x[1::])))
            edges = edges.explode('entity_id')
            edges = edges['entity_id'].to_list()

            G.add_edges_from(edges)
            # G.add_edges_from(chain.from_iterable(combinations(e, 2) for e in edges))
        # G = nx.compose_all(map(nx.complete_graph, edges))
        # G = nx.karate_club_graph()

        # SAME_CLUB_COLOR, DIFFERENT_CLUB_COLOR = "darkgrey", "red"
        edge_attrs = {}
        for start_node, end_node, _ in G.edges(data=True):
            edge_attrs[(start_node, end_node)] = "black"

        nx.set_edge_attributes(G, edge_attrs, "edge_color")

        tooltips = """
        <div>
            <span style="font-size: 14px; color: blue;">Entity ID = @{Entity ID} </span>
        </div>
        """
        detail = """
        <div>
            <span style="font-size: 12px; color: blue;">{feature}:</span> <br>
            <span style="font-size: 12px;">@{{{feature}}}</span>
        </div>
        """
        columns = list(chain.from_iterable(source.values()))
        tooltips += '\n'.join([detail.format(feature=feature) for feature in columns])



        # tooltips = """
        # <div>
        #     <span style="font-size: 12px; color: blue;">Entity ID: </span>
        #     <span style="font-size: 12px;">@{Entity ID}</span>
        # </div>
        # <div>
        #     <span style="font-size: 12px; color: blue;">ContactAddress: </span>
        #     <span style="font-size: 12px;">@{ContactAddress}</span>
        # </div>
        # """


        plot = figure(width=800, height=600, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
                    x_axis_location=None, y_axis_location=None,
                    title="Graph Interaction Demo", background_fill_color="#efefef",
                    tooltips=tooltips
        )
        plot.grid.grid_line_color = None

        graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="lightblue")
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
                                                    line_alpha=0.8, line_width=1.5)
        plot.renderers.append(graph_renderer)

        output_file(file_name+'.html')

        show(plot)