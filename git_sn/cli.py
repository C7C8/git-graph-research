"""The command line interface that drives the various scripts in this package"""
import argparse
import json
import sys
from typing import List
import networkx as nx
import pandas as pd
from networkx.algorithms.community import asyn_lpa_communities
from matplotlib import pyplot as plt

from . import __version__
from .graph import generate_author_graph, generate_file_graph, convert_to_json
from .parser import parse_raw_commits, Commit


def get_parser() -> argparse.ArgumentParser:
    """Get parser with all associated subparsers"""
    parser = argparse.ArgumentParser(description="Collection of scripts and commands for parsing, visualizing, and"
                                                 "analyzing Git histories as a social network", prog="GitSN")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-t", "--threshold", default=2, type=int,
                        help="Minimum edge weight to be considered when pruning")
    parser.add_argument("repo", type=str, help="Path to .git folder to analyze", default=".")
    parser.add_argument("type", type=str, choices=["commit", "file", "author"])
    subparsers = parser.add_subparsers(title="Operation", description="Operation to perform")

    graph_gen_parser = subparsers.add_parser("export", help="Data export operations")
    graph_gen_parser.add_argument("-o", "--out", type=argparse.FileType("w"), default=sys.stdout,
                                  help="File to write output to. Stdout by default.")
    graph_gen_parser.add_argument("-p", "--use-pandas", action="store_true",
                                  help="Export to NetworkX pandas format for file/author networks")
    graph_gen_parser.set_defaults(func=_export)

    visualize_parser = subparsers.add_parser("visualize", help="Visualization of graphs")
    visualize_parser.add_argument("-l", "--layout", type=str, choices=["kamada", "spring"], default="kamada",
                                  help="Type of layout to use when drawing")
    visualize_parser.set_defaults(func=_visualize)

    contact_parser = subparsers.add_parser("whodoitalkto")

    return parser


def _get_graph(args: argparse.Namespace, commits: List[Commit]):
    return (generate_file_graph if args.type == "file" else generate_author_graph)(commits, args.threshold)


@parse_raw_commits
def _export(args: argparse.Namespace, commits: List[Commit]):
    """Function for the export subcommand"""
    if args.type == "commit":
        json.dump(commits, args.out, indent=4)
        return

    graph = _get_graph(args, commits)
    if not args.use_pandas:
        json.dump(convert_to_json(graph), args.out, indent=4)
    else:
        pd.to_pickle(nx.to_pandas_edgelist(graph), args.out)


@parse_raw_commits
def _visualize(args: argparse.Namespace, commits: List[Commit]):
    """Function for the visualize subcommand"""
    graph = _get_graph(args, commits)

    if args.layout == "kamada":
        pos = nx.kamada_kawai_layout(graph, center=(0.5, 0.5), scale=0.5, weight="count")
    else:
        pos = nx.spring_layout(graph, center=(0.5, 0.5), scale=0.5, k=1/len(graph)**0.1, seed=1)

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


