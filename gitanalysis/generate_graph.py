#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from io import StringIO
from typing import Dict, IO

import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities


author_regex = re.compile("(?<=Author: ).*(?=\\s<)")

def parse_commits(file: IO):
	commits = split_commits(file)

	commits_dicts = []
	for commit in commits:
		cur = {"author": "", "files": []}
		line: str
		for line in commit:
			if line.startswith("Author: "):
				cur["author"] = author_regex.findall(line)[0]
				continue
			stripped = line.strip(" \n")
			if ":" in line or line.startswith(("commit", " ")) or len(stripped) == 0:
				continue
			cur["files"].append(stripped)

		commits_dicts.append(cur)

	return commits_dicts


def split_commits(file):
	"""Split one long file into a list of individual commit blocks"""
	ret = []
	cur = []
	for line in file:
		if line.startswith("commit") and len(cur) > 0:
			ret.append(cur)
			cur = []
		cur.append(line)
	return ret


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
	# df: pd.DataFrame = nx.to_pandas_edgelist(g)
	# df = df[df["count"] > threshold].sort_values("count", ascending=False)
	# return df


def _assign_groups(g: nx.Graph):
	"""Assign groups to nodes using the asynchronous LPA communities method"""
	for i, group in enumerate(asyn_lpa_communities(g, weight="count")):
		for node in group:
			g.nodes[node]["group"] = i


def convert_to_json(g: nx.Graph) -> Dict:
	"""Convert a graph to a JSON object that D3 can read more easily, with nodes and edges"""
	ret = {"nodes": [], "edges": []}
	for node in g.nodes:
		ret["nodes"].append({"id": node, "group": g.nodes[node]["group"]})
	for edge in g.edges:
		ret["edges"].append({"source": edge[0], "target": edge[1], "weight": g.edges[edge]["count"]})
	return ret


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("repo", type=str, help="Directory of git repository to process", nargs="*", default=".")
	parser.add_argument("--type", type=str, required=False, default="file", choices=["file", "author"], help="Type of graph to generate")
	parser.add_argument("--threshold", "-t", type=int, required=False, default=1, help="Minimum required weight for an edge to be included in the graph")
	args = parser.parse_args()

	try:
		commits = []
		for repo in args.repo:
			os.stat(repo)
			out = subprocess.run(["git", "--git-dir", repo, "log", "--name-only", "--stat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			if out.returncode > 0:
				print("git --git-dir {} log --name-only --stat failed".format(repo), file=sys.stderr)
				print(out.stderr.decode("utf-8"), file=sys.stderr)
			commits.extend(parse_commits(StringIO(out.stdout.decode("utf-8", errors="backslashreplace"))))
	except (FileNotFoundError, IOError):
		print("Repo not found or inaccessible".format(repo), file=sys.stderr)
		exit(1)

	if args.type == "file":
		graph = generate_file_graph(commits, args.threshold)
	else:
		graph = generate_author_graph(commits, args.threshold)

	json.dump(convert_to_json(graph), sys.stdout, indent=4)
