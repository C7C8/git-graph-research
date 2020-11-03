import itertools
import os
from typing import Dict

import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities


def generate_file_graph(commits, threshold=2) -> nx.Graph:
    """Generate a graph of files as connected """
    g = nx.Graph()
    # For every commit, link files that occur in the same commit
    for commit in commits:
        for file1, file2 in itertools.combinations(commit["files"], 2):
            # Add one to the count for this edge
            file1 = os.path.basename(file1)
            file2 = os.path.basename(file2)
            if g.has_edge(file1, file2):
                g.edges[file1, file2]["count"] += 1
            else:
                g.add_edge(file1, file2, count=1)

    # Do some filtering
    for edge in g.edges:
        if g.edges[edge]["count"] < threshold:
            g.remove_edge(*edge)
    for node in list(g.nodes):
        if node.endswith(".dll") or node.endswith(".dat") or node.endswith(".seg"):
            g.remove_node(node)
    _assign_groups(g)
    return g


def generate_author_graph(commits, threshold=1) -> nx.Graph:
    """Generate a graph of authors as connected by commits to files"""
    files = {}
    # Assemble a list of files and the authors who contributed to them
    for commit in commits:
        author = commit["author"]
        for file in commit["files"]:
            file = os.path.basename(file)
            if file not in files:
                files[file] = {author: 1}
            elif author not in files[file]:
                files[file][author] = 1
            else:
                files[file][author] += 1

    # For each file, calculate edges between people
    g = nx.Graph()
    for file in files.values():
        if len(file) == 1:
            continue

        for person1, person2 in itertools.combinations(file.keys(), 2):
            val = min(file[person1], file[person2])
            if g.has_edge(person1, person2):
                g.edges[person1, person2]["count"] += val
            else:
                g.add_edge(person1, person2, count=val)

    # Do some filtering
    for edge in g.edges:
        if g.edges[edge]["count"] < threshold:
            g.remove_edge(*edge)
    _assign_groups(g)
    return g


def _assign_groups(g: nx.Graph):
    """Assign groups to nodes using the asynchronous LPA communities method"""
    for i, group in enumerate(asyn_lpa_communities(g, weight="count")):
        for node in group:
            g.nodes[node]["group"] = i


def convert_to_json(g: nx.Graph) -> Dict:
    """Convert a graph to a JSON object that's more human readable, with nodes and edges"""
    ret = {"nodes": [], "edges": []}
    for node in g.nodes:
        ret["nodes"].append({"id": node, "group": g.nodes[node]["group"]})
    for edge in g.edges:
        ret["edges"].append({"source": edge[0], "target": edge[1], "weight": g.edges[edge]["count"]})
    return ret
