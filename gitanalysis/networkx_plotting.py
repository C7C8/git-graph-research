import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms.community import asyn_lpa_communities

from gitanalysis.generate_graph import generate_file_graph, parse_commits

matplotlib.rcParams['figure.dpi'] = 900

commits = parse_commits("linux.txt")
graph = generate_file_graph(commits)

# df: pd.DataFrame = generate_author_edgelist(commits, 10)

# graph = nx.from_pandas_edgelist(df, "source", "target", ["count"])
# pos = nx.spring_layout(graph, center=(0.5, 0.5), scale=0.5, k=1/len(graph)**0.1, seed=1)
pos = nx.kamada_kawai_layout(graph, center=(0.5, 0.5), scale=0.5, weight="count")

# Base edges
nx.draw_networkx_edges(graph, pos=pos, width=0.1, alpha=0.3)

# Bridges
if type(graph) is not nx.DiGraph and nx.has_bridges(graph):
	nx.draw_networkx_edges(graph, edgelist=list(nx.bridges(graph)), pos=pos, width=0.2, alpha=0.5, edge_color="r")

# Nodes with colors
groups = list(asyn_lpa_communities(graph, weight="count"))
colors = [[file in group for group in groups].index(True) for file in graph]
nx.draw_networkx_nodes(graph, pos=pos, node_color=colors, cmap="tab20", node_size=5)

# Labels
nx.draw_networkx_labels(graph, pos, font_size=1)

plt.show()
