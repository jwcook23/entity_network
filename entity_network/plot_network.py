from itertools import chain

import networkx as nx

from bokeh.models import Circle, MultiLine, HoverTool
from bokeh.plotting import figure, from_networkx, show, output_file

# http://docs.bokeh.org/en/latest/docs/gallery/network_graph.html
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html


def plot_network(graph):

    plot = figure(width=400, height=400, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
        x_axis_location=None, y_axis_location=None,
        title="Graph Interaction Demo", background_fill_color="#efefef",
    )
    plot.grid.grid_line_color = None

    node_details = [graph.nodes[x].keys() for x in graph.nodes]
    node_details = set(chain(*node_details))
    tooltips = [(x,f'@{x}') for x in node_details]
    node_hover_tool = HoverTool(tooltips=tooltips)
    plot.add_tools(node_hover_tool)

    graph_renderer = from_networkx(graph, nx.spring_layout, scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="lightblue")
    # graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
    #                                             line_alpha=0.8, line_width=1.5)
    plot.renderers.append(graph_renderer)

    output_file('NetworkGraph.html')

    show(plot)