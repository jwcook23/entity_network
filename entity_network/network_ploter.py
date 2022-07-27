# http://docs.bokeh.org/en/latest/docs/gallery/network_graph.html
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html

from itertools import combinations, chain
from collections import OrderedDict

import networkx as nx
import pandas as pd
from tornado.ioloop import IOLoop
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.models import Circle, MultiLine
from bokeh.plotting import figure, from_networkx
from bokeh.server.server import Server

class server():

    def __init__(self, network_map, network_feature, entity):
        
        self.network_map = network_map
        self.network_feature = network_feature
        self.entity = entity
        
        self.summerize_network()

        # track a single network only, defaulting to the first
        self.network_map = self.network_map[self.network_map['network_id']==self.network_summary.index[0]]

        # remove duplicates for an entity
        self.network_map = self.network_map.drop_duplicates(subset=['entity_id','address_id','phone_id','email_id'])

        self.main()

    def main(self):

        io_loop = IOLoop.current()
        bokeh_app = Application(FunctionHandler(self.modify_doc))

        server = Server({"/": bokeh_app}, io_loop=io_loop)
        server.start()
        print("Opening Bokeh application on http://localhost:5006/")

        io_loop.add_callback(server.show, "/")
        io_loop.start()


    def modify_doc(self, doc):

        G = nx.Graph()

        self.node_details(G)

        self.edge_details(G)

        tooltips = self.generate_tooltip()

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

        doc.add_root(plot)
        doc.title = "Test Plot"


    def summerize_network(self):

        # TODO: network_summary should be part of base er package
        network_summary = self.network_map.groupby('network_id')
        network_summary = network_summary.agg({'entity_id': 'nunique'})
        network_summary = network_summary.rename(columns={'entity_id': 'entity_count'})
        network_summary = network_summary.sort_values('entity_count', ascending=False)

        self.network_summary = network_summary


    def node_details(self, G):

        entity = self.entity.groupby(['entity_id', 'column'])
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


    def edge_details(self, G):
        # add edges
        for category in self.network_feature.keys():
            if category=='name':
                continue
            edges = self.network_map.groupby(f'{category}_id')
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


    def generate_tooltip(self):

        # determine unique columns and ensure name is first for display purposes
        source = list(self.network_feature.keys())
        source.remove('name')
        source = ['name']+source
        source = OrderedDict(zip(source, [None]*len(source)))
        for category in source.keys():
            source[category] = self.entity.loc[self.entity['category']==category,'column'].unique()

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

        return tooltips
