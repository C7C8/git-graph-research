"""Git log parser functions and useful annotations"""
import argparse
import os
import re
import subprocess
import sys
from io import StringIO
from typing import IO, List, Dict, Union, Callable, NoReturn

author_regex = re.compile("(?<=Author: ).*(?=\\s<)")
Commit = Dict[str, Union[str, List[str]]]


def parse_commits(file: IO) -> List[Commit]:
    """Parse commits from a file into a list of dicts, one for each commit."""
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


def split_commits(file: IO) -> List[List[str]]:
    """Split one long file into a list of individual commit blocks"""
    ret = []
    cur = []
    for line in file:
        if line.startswith("commit") and len(cur) > 0:
            ret.append(cur)
            cur = []
        cur.append(line)
    return ret


# Annotation for allowing any function to accept a list repos and get a list of raw commits
def parse_raw_commits(func: Callable[[argparse.Namespace, List[Commit]], NoReturn]):
    def decorator_wrapper(args):
        repo = args.repo
        commits = []
        try:
            os.stat(repo)
            out = subprocess.run(["git", "--git-dir", repo, "log", "--name-only", "--stat"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            if out.returncode > 0:
                print("git --git-dir {} log --name-only --stat failed".format(repo), file=sys.stderr)
                print(out.stderr.decode("utf-8"), file=sys.stderr)
            commits.extend(parse_commits(StringIO(out.stdout.decode("utf-8", errors="backslashreplace"))))
            func(args, commits)
        except (FileNotFoundError, IOError):
            print("Repo '{}' not found or inaccessible".format(repo), file=sys.stderr)
            exit(1)
    return decorator_wrapper
