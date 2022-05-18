import sample

from entity_network.entity_resolver import entity_resolver

import networkx as nx

from bokeh.models import Circle, MultiLine, HoverTool
from bokeh.plotting import figure, from_networkx, show

# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html

n_unique = 1000
n_duplicates = 30

# generate sample data
sample_df = sample.unique_records(n_unique)
columns = {'phone': ['HomePhone','WorkPhone','CellPhone']}
sample_df, sample_id, sample_feature = sample.duplicate_records(sample_df, n_duplicates, columns)

# compare and derive network
er = entity_resolver(sample_df)
er.compare('phone', columns=columns['phone'])
network_id, network_feature = er.network()

G = nx.karate_club_graph()
for start_node, end_node, _ in G.edges(data=True):
    pass

G = er.network_graph
plot = figure(width=400, height=400, x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
            x_axis_location=None, y_axis_location=None,
            title="Graph Interaction Demo", background_fill_color="#efefef",
        #   tooltips="index: @index, club: @club"
)
plot.grid.grid_line_color = None

# node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("name", "@club")])

graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="lightblue")
# graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color",
#                                             line_alpha=0.8, line_width=1.5)
plot.renderers.append(graph_renderer)

show(plot)