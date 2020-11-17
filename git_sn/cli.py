"""The command line interface that drives the various scripts in this package"""
import argparse
import inspect
import json
import sys
from typing import List, Dict

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx.algorithms.community import asyn_lpa_communities

from . import __version__
from .graph import generate_author_graph, generate_file_graph, convert_to_json, generate_bi_graph
from .parser import parse_raw_commits, Commit

centrality_algo_dict = {
    "pagerank": nx.pagerank,
    "degree": nx.degree_centrality,
    "eigenvector": nx.eigenvector_centrality,
    "katz": nx.katz_centrality,
    "closeness": nx.closeness_centrality,
    "current_flow_closeness": nx.current_flow_betweenness_centrality,
    "information": nx.information_centrality,
    "betweenness": nx.betweenness_centrality,
    "betweenness_source": nx.betweenness_centrality_source,
    "betweenness_edge": nx.edge_betweenness_centrality,
    "betweenness_current_flow": nx.current_flow_betweenness_centrality,
    "edge_current_flow_betweenness": nx.edge_current_flow_betweenness_centrality,
    "communicability": nx.communicability_betweenness_centrality,
    "load": nx.load_centrality,
    "subgraph": nx.subgraph_centrality,
    "subgraph_centrality_exp": nx.subgraph_centrality_exp,
    "estrada": nx.estrada_index,
    "harmonic": nx.harmonic_centrality,
    "global_reaching": nx.global_reaching_centrality,
    "second_order": nx.second_order_centrality,
    "voterank": nx.voterank
}


def get_parser() -> argparse.ArgumentParser:
    """Get parser with all associated subparsers"""
    parser = argparse.ArgumentParser(description="Collection of scripts and commands for parsing, visualizing, and"
                                                 "analyzing Git histories as a social network", prog="GitSN")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-t", "--threshold", default=2, type=int,
                        help="Minimum edge weight to be considered when pruning")
    parser.add_argument("repo", type=str, help="Path to .git folder to analyze", default=".")
    parser.add_argument("type", type=str, choices=["bi", "file", "author"], help="Type of data format to generate")
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

    rank_parser = subparsers.add_parser("rank", help="Centrality ranking of data to find the top N most important items")
    rank_parser.add_argument("algorithm", type=str, help="Centrality algorithm to use", choices=centrality_algo_dict.keys())
    rank_parser.add_argument("-n", "--number", help="Number of top items to print. If 0, prints all", type=int, default=10)
    rank_parser.add_argument("-a", "--alpha", help="Alpha factor for various algorithms", type=float, default=0.85)
    rank_parser.add_argument("-b", "--beta", help="Beta factor for various algorithms", type=float, default=1)
    rank_parser.add_argument("-m", "--max-iter", help="Max iterations for iterative-based solvers", type=int, default=1000)
    rank_parser.set_defaults(func=_rank)

    contact_parser = subparsers.add_parser("whodoitalkto", help="Find appropriate neighbors on the graph given a single target")
    contact_parser.add_argument("target", type=str, help="Target to generate advice for")
    contact_parser.add_argument("-n", "--number", type=int, help="Number of results N to return", default=3)
    contact_parser.add_argument("-d", "--depth", type=int, default=1, help="Connection depth D to search to")
    contact_parser.add_argument("-m", "--method", type=str, choices=["nearest", "bfs-terminal", "next-community"],
                                  help="""Algorithm to use when searching. Nearest mode will take all direct
                                  neighbors to the target, sort them by connection strength, and return the top N of
                                  those. BFS-terminal will execute a breadth-first search to the target depth, then choose
                                  the top N nodes with the strongest connection to the target. Next-community mode will
                                  act similar to BFS-terminal, but will start from the bounds of the community instead.""")
    contact_parser.set_defaults(func=_whodoitalkto)

    return parser


def _get_graph(args: argparse.Namespace, commits: List[Commit]):
    if args.type == "file":
        return generate_file_graph(commits, args.threshold)
    elif args.type == "author":
        return generate_author_graph(commits, args.threshold)
    elif args.type == "bi":
        return generate_bi_graph(commits, args.threshold)


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
    plt.figure(figsize=(18, 18))
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


@parse_raw_commits
def _whodoitalkto(args: argparse.Namespace, commits: List[Commit]):
    """Function for the whodoitalkto subcommand"""
    graph = _get_graph(args, commits)

    if args.target not in graph.nodes:
        print("Target '{}' not found in graph".format(args.target), file=sys.stderr)
        exit(1)


def _filter_dict(filterable: Dict, func):
    """For a dict of kwargs and a given function, filter the kwargs to kwargs the function is capable of accepting"""
    sig = inspect.signature(func)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filter_keys = filter(lambda key: key in filterable.keys(), filter_keys)
    filtered_dict = {filter_key: filterable[filter_key] for filter_key in filter_keys}
    return filtered_dict


@parse_raw_commits
def _rank(args: argparse.Namespace, commits: List[Commit]):
    """Function for ranking subcommand"""
    graph = _get_graph(args, commits)
    algo = centrality_algo_dict[args.algorithm]
    kwargs = _filter_dict({
        "alpha": args.alpha,
        "beta": args.beta,
        "weight": "count",
        "max_iter": args.max_iter
    }, algo)

    result = algo(graph, **kwargs)
    if isinstance(result, dict):
        ranking = list(sorted(result.items(), key=lambda item: item[1], reverse=True))
        for rank, node in enumerate(ranking):
            if rank >= args.number:
                break
            print("{rank}. {name} ({val})".format(rank=rank + 0, name=node[0], val=node[1]))
    elif isinstance(result, float):
        print(result)

