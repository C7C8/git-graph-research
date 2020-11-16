#!/usr/bin/env zsh

# This script is old and needs to be updated to the new CLI
mkdir -p networks/file networks/author
parallel --jobs 8 --eta --progress --verbose ./generate_graph.py --type author -t 2 {}/.git ">" networks/author/{/}.json ::: `find repos -maxdepth 1 -mindepth 1 -not -name ".idea"`
parallel --jobs 8 --eta --progress --verbose ./generate_graph.py --type file -t 2 {}/.git ">" networks/file/{/}.json ::: `find repos -maxdepth 1 -mindepth 1 -not -name ".idea"`
