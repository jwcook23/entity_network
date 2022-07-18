'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations, chain

import pandas as pd
import networkx as nx
from bokeh.models import Circle, MultiLine, HoverTool
from bokeh.plotting import figure, from_networkx, show, output_file

from entity_network import _index, _prepare, _compare, _helpers, _exceptions

class entity_resolver():


    def __init__(self, df:pd.DataFrame, df2:pd.DataFrame = None):

        self._df, self._index_mask = _index.unique(df, df2)

        self.network_feature = {}
        self.similar_score = {}
        # self.network_map = None
        # self.network_id = None
        # self.entity_feature = None
        # self.entity_id = None
        self.processed = {}
        self.timer = pd.DataFrame(columns=['caller','file','method','category','time_seconds'])


    def compare(self, category, columns, kneighbors:int=10, threshold:float=1, text_comparer='default', text_cleaner='default'):

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

        # ignore values the processor completely removed
        # self.processed[category] = self.processed[category].dropna()

        # # compare values on similarity threshold
        # print(f'Comparing category: {category}.')
        # tstart = time()
        # self.network_feature[category], self.similar_score[category] = _compare.match(category, self.processed[category], kneighbors, threshold, text_comparer, self._index_mask)
        # self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

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

        # assign class properties
        self.network_feature[category] = related_feature
        self.similar_score[category] = similar_score

        # add original index to processed values
        self.processed[category]['df'] = self.processed[category]['df'].reset_index()
        self.processed[category]['df'] = self.processed[category]['df'].merge(self._index_mask['df'], on='node')
        if self._index_mask['df2']is not None:
            self.processed[category]['df2'] = self.processed[category]['df2'].reset_index()
            self.processed[category]['df2'] = self.processed[category]['df2'].merge(self._index_mask['df2'], on='node')

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)


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

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)

        return self.network_id, self.network_map, self.network_feature


    def debug_similar(self, category):
        
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

        # determine differences between similar values
        # TODO: provide a summary of differences
        columns = list(df_categories.values())+list(df2_categories.values())
        columns = [x for x in columns if x is not None]
        similar[f'{category}_difference'] = similar[columns].apply(comparer[category], axis=1)

        # group by id and score
        similar = similar.sort_values(by=['id_similar','score'], ascending=[True, False])

        # split by in/out of cluster with closest distance to cluster edge appearing first
        in_cluster = similar[similar['threshold']].sort_values(by='score', ascending=True)
        out_cluster = similar[~similar['threshold']].sort_values(by='score', ascending=False)

        return similar, in_cluster, out_cluster

    def plot_network(self, file_name):
    # http://docs.bokeh.org/en/latest/docs/gallery/network_graph.html
    # https://docs.bokeh.org/en/latest/docs/user_guide/graph.html

        # form nodes using unique pairwise connections
        edges = self.network_id.reset_index()
        edges = edges.groupby('network_id')
        edges = edges.agg({'node': list})
        edges = edges['node'].to_list()
        # G = nx.from_edgelist(chain.from_iterable(combinations(e, 2) for e in edges))
        G = nx.compose_all(map(nx.complete_graph, edges))
        # G = nx.karate_club_graph()

        # SAME_CLUB_COLOR, DIFFERENT_CLUB_COLOR = "darkgrey", "red"
        edge_attrs = {}

        for start_node, end_node, _ in G.edges(data=True):
            # edge_color = SAME_CLUB_COLOR if G.nodes[start_node]["club"] == G.nodes[end_node]["club"] else DIFFERENT_CLUB_COLOR
            edge_attrs[(start_node, end_node)] = "black"

        nx.set_edge_attributes(G, edge_attrs, "edge_color")

        plot = figure(width=800, height=600, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
                    x_axis_location=None, y_axis_location=None,
                    title="Graph Interaction Demo", background_fill_color="#efefef",
                    tooltips="index: @index, club: @club")
        plot.grid.grid_line_color = None

        graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="lightblue")
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
                                                    line_alpha=0.8, line_width=1.5)
        plot.renderers.append(graph_renderer)

        output_file(file_name+'.html')

        show(plot)