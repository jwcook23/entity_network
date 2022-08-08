# http://docs.bokeh.org/en/latest/docs/gallery/network_graph.html
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html

from itertools import combinations, chain

import networkx as nx
from tornado.ioloop import IOLoop
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.models import Circle, MultiLine
from bokeh.plotting import figure, from_networkx
from bokeh.server.server import Server
from bokeh.layouts import row, column

class network_dashboard():

    def __init__(self):

        self.network_selected = None
        self.network_selected = self.network_summary.index[0]

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

        network = self.plot_graph()
        features = self.plot_features()

        layout = network

        doc.add_root(layout)
        doc.title = "Network Dashboard"


    def plot_graph(self):

        G = nx.Graph()
        self.node_details(G)
        self.edge_details(G)

        tooltips = self.generate_tooltip()

        plot = figure(width=800, height=600, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
                    x_axis_location=None, y_axis_location=None,
                    title="Selected Network Graph", background_fill_color="#efefef",
                    tooltips=tooltips
        )
        plot.grid.grid_line_color = None

        graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="lightblue")
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
                                                    line_alpha=0.8, line_width=1.5)
        plot.renderers.append(graph_renderer)

        return plot

    def plot_features(self):

        # self.entity[self.entity['network_id']==self.network_selected]
        feature = None
        return feature

    def node_details(self, G):

        entity = self.entity[self.entity['network_id']==self.network_selected]
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


    def edge_details(self, G):
        # add edges
        for category in self.network_feature.keys():
            if category=='name':
                continue

            edges = self.network_map[self.network_map['network_id']==self.network_selected]
            edges = edges.drop_duplicates(subset=['entity_id','address_id','phone_id','email_id'])
            edges = edges.groupby(f'{category}_id')
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
        columns = list(chain.from_iterable(self._compared_columns.values()))
        tooltips += '\n'.join([detail.format(feature=feature) for feature in columns])

        return tooltips
