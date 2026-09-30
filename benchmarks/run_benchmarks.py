"""Benchmark harness: times BST, AVL, Red-Black and Splay trees.

Two experiments, each writing one CSV to results/ (one row per run):

1. random.csv -- for Figures 1-5 and the main performance table.
   For every size n: build the tree from the shuffled keys, then time
   uniform searches and deletions, and record height, rotations and memory.

2. workloads.csv -- for the workload-comparison table (default n = 10^6).
   Builds from random, sorted-ascending and sorted-descending input, and
   compares uniform searches with hot-key (skewed) searches.
   Sorted input makes a plain BST quadratic, so the BST's sorted runs use
   only the first --bst-sorted-cap keys; the CSV's n_used column records this.
"""

import argparse
import csv
import gc
import os
import time
import tracemalloc
from contextlib import contextmanager

from benchmarks.generate_data import (
    DATA_DIR, DEFAULT_SIZES, dataset_path, load_dataset, save_dataset,
)
from trees.avl import AVLTree
from trees.bst import BST
from trees.red_black import RedBlackTree
from trees.splay import SplayTree

STRUCTURES = {"BST": BST, "AVL": AVLTree, "Red-Black": RedBlackTree, "Splay": SplayTree}
RESULTS_DIR = "results"

RANDOM_FIELDS = ["n", "structure", "run", "build_s", "search_us", "delete_us",
                 "memory_mb", "height", "rotations"]
WORKLOAD_FIELDS = ["workload", "structure", "run", "n_used", "build_s",
                   "search_us", "height"]


# ---------- timing helpers ----------

@contextmanager
def gc_paused():
    """
    Input: none (used as `with gc_paused(): ...`).
    Output: None.
    Collects garbage, then disables the garbage collector for the block so
    it cannot interrupt timing, and re-enables it afterwards.
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def build_tree(tree_class, keys):
    """
    Input: tree_class -- one of the classes in STRUCTURES; keys -- list of ints.
    Output: (tree, seconds) -- the built tree and the time taken to build it.
    """
    tree = tree_class()
    insert = tree.insert  # local name avoids repeated attribute lookups
    with gc_paused():
        start = time.perf_counter()
        for key in keys:
            insert(key)
        seconds = time.perf_counter() - start
    return tree, seconds


def time_per_operation(operation, keys):
    """
    Input: operation -- a bound method such as tree.search; keys -- list of ints.
    Output: mean time per call in microseconds (0.0 if keys is empty).
    """
    if not keys:
        return 0.0
    with gc_paused():
        start = time.perf_counter()
        for key in keys:
            operation(key)
        seconds = time.perf_counter() - start
    return seconds / len(keys) * 1e6


def measure_memory_mb(tree_class, keys):
    """
    Input: tree_class -- one of the classes in STRUCTURES; keys -- list of ints.
    Output: memory allocated while building the tree, in MB.
    The keys already exist before tracing starts, so this counts the tree's
    own nodes and bookkeeping, not the integers themselves.
    """
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tree = tree_class()
    for key in keys:
        tree.insert(key)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    del tree
    gc.collect()
    return (after - before) / 1e6


# ---------- data and output helpers ----------

def get_dataset(n, data_dir=DATA_DIR):
    """
    Input: n -- dataset size; data_dir -- dataset folder.
    Output: the dataset for size n as a dict of lists (generated if missing).
    """
    if not os.path.exists(dataset_path(n, data_dir)):
        print(f"  dataset for n = {n:,} not found; generating it...", flush=True)
        save_dataset(n, data_dir)
    return load_dataset(n, data_dir)


def open_csv(path, fields):
    """
    Input: path -- CSV file to create (overwritten); fields -- column names.
    Output: (file, csv.DictWriter) with the header already written.
    Rows are flushed as they are written, so partial results survive a crash.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handle = open(path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


# ---------- experiments ----------

def run_random_experiment(sizes, runs, out_path, data_dir=DATA_DIR, with_memory=True):
    """
    Input: sizes -- list of n values; runs -- repetitions per (n, structure);
           out_path -- CSV path; data_dir -- dataset folder;
           with_memory -- measure memory (slower) or leave it blank.
    Output: None (writes one CSV row per n, structure and run).
    Order per run: build -> record height -> uniform searches -> deletions.
    """
    handle, writer = open_csv(out_path, RANDOM_FIELDS)
    with handle:
        for n in sizes:
            data = get_dataset(n, data_dir)
            for name, tree_class in STRUCTURES.items():
                memory = measure_memory_mb(tree_class, data["keys"]) if with_memory else ""
                for run in range(1, runs + 1):
                    tree, build_s = build_tree(tree_class, data["keys"])
                    height, rotations = tree.height(), tree.rotations
                    search_us = time_per_operation(tree.search, data["uniform_queries"])
                    delete_us = time_per_operation(tree.delete, data["delete_keys"])
                    writer.writerow({
                        "n": n, "structure": name, "run": run,
                        "build_s": f"{build_s:.6f}", "search_us": f"{search_us:.4f}",
                        "delete_us": f"{delete_us:.4f}",
                        "memory_mb": f"{memory:.4f}" if memory != "" else "",
                        "height": height, "rotations": rotations,
                    })
                    handle.flush()
                    del tree  # free it before the next build (trees can use GBs)
                    gc.collect()
                    print(f"  random  n={n:>10,}  {name:<9}  run {run}: "
                          f"build {build_s:8.3f}s  search {search_us:7.2f}us  "
                          f"delete {delete_us:7.2f}us  height {height}", flush=True)


def run_workload_experiment(n, runs, bst_sorted_cap, out_path, data_dir=DATA_DIR):
    """
    Input: n -- dataset size; runs -- repetitions; bst_sorted_cap -- maximum
           number of keys for the BST's sorted runs; out_path -- CSV path;
           data_dir -- dataset folder.
    Output: None (writes one CSV row per workload, structure and run).
    Workloads:
      random / sorted ascending / sorted descending -- build from that order,
          record height, then time uniform searches on the result;
      hot-key search -- build from random order, then time hot_queries.
    """
    data = get_dataset(n, data_dir)
    keys, uniform, hot = data["keys"], data["uniform_queries"], data["hot_queries"]

    handle, writer = open_csv(out_path, WORKLOAD_FIELDS)
    with handle:
        for name, tree_class in STRUCTURES.items():
            for run in range(1, runs + 1):
                cases = [("Random", keys, uniform)]
                for label, reverse in [("Sorted ascending", False),
                                       ("Sorted descending", True)]:
                    if name == "BST" and n > bst_sorted_cap:
                        subset = keys[:bst_sorted_cap]    # random subset of keys
                        queries = subset[:len(uniform)]   # searches that hit it
                    else:
                        subset, queries = keys, uniform
                    cases.append((label, sorted(subset, reverse=reverse), queries))

                for label, insert_order, queries in cases:
                    tree, build_s = build_tree(tree_class, insert_order)
                    height = tree.height()
                    search_us = time_per_operation(tree.search, queries)
                    writer.writerow({
                        "workload": label, "structure": name, "run": run,
                        "n_used": len(insert_order), "build_s": f"{build_s:.6f}",
                        "search_us": f"{search_us:.4f}", "height": height,
                    })
                    handle.flush()
                    del tree  # free it before the next build (trees can use GBs)
                    gc.collect()
                    print(f"  workload {label:<18} {name:<9} run {run}: "
                          f"n={len(insert_order):,}  build {build_s:8.3f}s  "
                          f"search {search_us:8.2f}us  height {height}", flush=True)

                tree, _ = build_tree(tree_class, keys)
                search_us = time_per_operation(tree.search, hot)
                writer.writerow({
                    "workload": "Hot-key search", "structure": name, "run": run,
                    "n_used": n, "build_s": "", "search_us": f"{search_us:.4f}",
                    "height": "",
                })
                handle.flush()
                del tree
                gc.collect()
                print(f"  workload {'Hot-key search':<18} {name:<9} run {run}: "
                      f"search {search_us:8.2f}us", flush=True)


def main():
    """Parses command-line options and runs the selected experiments."""
    parser = argparse.ArgumentParser(description="Benchmark the four tree structures.")
    parser.add_argument("--experiment", choices=["random", "workloads", "all"],
                        default="all", help="which experiment(s) to run")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="n values for the random experiment")
    parser.add_argument("--runs", type=int, default=3,
                        help="repetitions per measurement (default 3)")
    parser.add_argument("--workload-n", type=int, default=10**6,
                        help="n for the workload experiment (default 10^6)")
    parser.add_argument("--bst-sorted-cap", type=int, default=10_000,
                        help="max keys for BST sorted-input runs (default 10,000)")
    parser.add_argument("--skip-memory", action="store_true",
                        help="skip memory measurement (faster)")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.experiment in ("random", "all"):
        print("Random experiment", flush=True)
        run_random_experiment(args.sizes, args.runs,
                              os.path.join(RESULTS_DIR, "random.csv"),
                              with_memory=not args.skip_memory)
    if args.experiment in ("workloads", "all"):
        print("Workload experiment", flush=True)
        run_workload_experiment(args.workload_n, args.runs, args.bst_sorted_cap,
                                os.path.join(RESULTS_DIR, "workloads.csv"))
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} minutes. "
          f"Results are in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()