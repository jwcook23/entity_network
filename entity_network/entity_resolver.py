'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
from time import time
from itertools import combinations, chain

import pandas as pd
import networkx as nx
from bokeh.models import Circle, MultiLine, HoverTool
from bokeh.plotting import figure, from_networkx, show, output_file

from entity_network import _index, _prepare, _compare, _helpers

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

        # combine split columns, flatten into single
        print(f'flattening {columns}')
        tstart = time()
        self.processed[category] = _prepare.flatten(self._df, columns)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'flatten', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # clean column text
        if text_cleaner is not None:
            print(f'cleaning {category}')
            tstart = time()
            self.processed[category] = _prepare.clean(self.processed[category], category, text_cleaner)
            self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_prepare', 'clean', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # set sources for tracability when comparing multiple categories
        self.processed[category].index.names = ('node', 'column')
        self.processed[category].name = category

        # ignore values the processor completely removed
        # self.processed[category] = self.processed[category].dropna()

        # compare values on similarity threshold
        print(f'comparing {columns}')
        tstart = time()
        self.network_feature[category], self.similar_score[category] = _compare.match(category, self.processed[category], kneighbors, threshold, text_comparer, self._index_mask)
        self.timer = pd.concat([self.timer, pd.DataFrame([['compare', '_compare', 'match', category, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # add original index to processed values
        self.processed[category] = self.processed[category].reset_index()
        self.processed[category] = _index.original(self.processed[category], self._index_mask)

        # sort by most time intensive
        self.timer = self.timer.sort_values(by='time_seconds', ascending=False)


    def network(self):

        print('forming network')

        # form matrix of indices connected on any feature
        tstart = time()
        network_map = _helpers.combine_features(self.network_feature, self._df.index)
        self.timer = pd.concat([self.timer, pd.DataFrame([['network', '_helpers', 'combine_features', None, time()-tstart]], columns=self.timer.columns)], ignore_index=True)

        # determine an overall id using indices connected on any feature
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
        processed = self.processed[category][['df_index', category]].dropna(subset='df_index')
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