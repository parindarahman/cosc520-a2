"""Synthetic dataset generator for the BST / AVL / Red-Black / Splay benchmarks.

For each size n, one .npz file is written to data/ containing:
    keys            -- n distinct even integers in random order (insertion order
                       for the random workload; sort them for the sorted workload)
    uniform_queries -- search keys drawn uniformly from `keys` (successful searches)
    miss_queries    -- odd integers, so never in the tree (unsuccessful searches)
    hot_queries     -- skewed searches: HOT_QUERY_SHARE of them hit a small
                       "hot" subset (HOT_KEY_FRACTION of the keys)
    delete_keys     -- distinct keys from `keys`, to be removed

Every file is reproducible from SEED, so re-running the script regenerates
exactly the same data.

CSV copies (one single-column file per array, under data/csv/n_<size>/) can
also be written with --csv; 
Usage (from the project root):
    python -m benchmarks.generate_data                  # all default sizes
    python -m benchmarks.generate_data --csv            # also write CSV copies
    python -m benchmarks.generate_data --sizes 1000 10000
"""

import argparse
import os

import numpy as np

SEED = 520                   # base random seed (course number)
DEFAULT_SIZES = [10**3, 10**4, 10**5, 10**6, 10**7]
MAX_QUERIES = 100_000        # cap on queries/deletes timed per experiment
HOT_KEY_FRACTION = 0.01      # 1% of keys are "hot"
HOT_QUERY_SHARE = 0.90       # 90% of hot-workload queries target hot keys
DATA_DIR = "data"


def generate_dataset(n, seed=SEED):
    """
    Input: n -- number of keys (must be >= 1); seed -- base random seed.
    Output: dict mapping array names to int64 numpy arrays (see module docstring).
    Builds all key and query arrays for one dataset size. The generator is
    seeded with (seed, n) so each size is reproducible independently.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng([seed, n])
    num_queries = min(n, MAX_QUERIES)

    # Even numbers 0, 2, ..., 2(n-1) in random order: distinct by construction.
    keys = 2 * rng.permutation(n).astype(np.int64)

    uniform_queries = rng.choice(keys, size=num_queries, replace=True)

    # Odd numbers lie strictly between stored keys, so every miss searches to a leaf.
    miss_queries = 2 * rng.integers(0, n, size=num_queries, dtype=np.int64) + 1

    # keys is already shuffled, so its first slice is a random hot subset.
    hot_count = max(1, int(n * HOT_KEY_FRACTION))
    hot_keys = keys[:hot_count]
    use_hot = rng.random(num_queries) < HOT_QUERY_SHARE
    hot_queries = np.where(use_hot,
                           rng.choice(hot_keys, size=num_queries),
                           rng.choice(keys, size=num_queries))

    delete_keys = rng.choice(keys, size=num_queries, replace=False)

    return {
        "keys": keys,
        "uniform_queries": uniform_queries,
        "miss_queries": miss_queries,
        "hot_queries": hot_queries,
        "delete_keys": delete_keys,
    }


def dataset_path(n, data_dir=DATA_DIR):
    """
    Input: n -- dataset size; data_dir -- folder holding the datasets.
    Output: file path string, e.g. data/dataset_1000.npz.
    """
    return os.path.join(data_dir, f"dataset_{n}.npz")


def csv_folder(n, data_dir=DATA_DIR):
    """
    Input: n -- dataset size; data_dir -- folder holding the datasets.
    Output: folder path string for this size's CSV files, e.g. data/csv/n_1000.
    """
    return os.path.join(data_dir, "csv", f"n_{n}")


def save_csv(dataset, n, data_dir=DATA_DIR):
    """
    Input: dataset -- dict of arrays from generate_dataset(); n -- dataset size;
           data_dir -- output folder.
    Output: path of the folder holding the CSV files.
    Writes one single-column CSV per array (keys.csv, uniform_queries.csv, ...),
    each with a header row naming the column. These copies are for sharing and
    inspection only; the benchmark reads the faster .npz file.
    """
    folder = csv_folder(n, data_dir)
    os.makedirs(folder, exist_ok=True)
    for name, values in dataset.items():
        np.savetxt(os.path.join(folder, f"{name}.csv"), values,
                   fmt="%d", header=name, comments="")
    return folder


def save_dataset(n, data_dir=DATA_DIR, seed=SEED, write_csv=False):
    """
    Input: n -- dataset size; data_dir -- output folder; seed -- base seed;
           write_csv -- if True, also write CSV copies (see save_csv).
    Output: path of the written .npz file.
    Generates the dataset for size n and writes it to disk.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset = generate_dataset(n, seed)
    path = dataset_path(n, data_dir)
    np.savez(path, **dataset)
    if write_csv:
        save_csv(dataset, n, data_dir)
    return path


def load_dataset(n, data_dir=DATA_DIR):
    """
    Input: n -- dataset size; data_dir -- folder holding the datasets.
    Output: dict mapping array names to plain Python lists of ints.
    Lists of Python ints are returned (not numpy arrays) because tree code
    compares keys one at a time, and numpy scalar comparisons are much slower.
    """
    with np.load(dataset_path(n, data_dir)) as data:
        return {name: data[name].tolist() for name in data.files}


def main():
    """Parses command-line sizes and writes one dataset file per size."""
    parser = argparse.ArgumentParser(description="Generate benchmark datasets.")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="dataset sizes to generate (default: 10^3 ... 10^7)")
    parser.add_argument("--csv", action="store_true",
                        help="also write CSV copies to data/csv/ for sharing")
    args = parser.parse_args()

    for n in args.sizes:
        path = save_dataset(n, write_csv=args.csv)
        size_mb = os.path.getsize(path) / 1e6
        print(f"n = {n:>10,}  ->  {path}  ({size_mb:.1f} MB)")
        if args.csv:
            print(f"{'':16}  ->  {csv_folder(n)}{os.sep}*.csv")


if __name__ == "__main__":
    main()