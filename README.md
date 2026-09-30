# COSC 520 Assignment 2: Balanced Binary Search Trees

Implementation and benchmarking of four binary search tree variants for COSC 520 (Advanced Data Structures).

| Structure | Role | Balancing strategy |
|---|---|---|
| Binary Search Tree (BST) | Baseline | None |
| AVL Tree | Advanced | Strict height balance (balance factor in {−1, 0, 1}) |
| Red-Black Tree | Advanced | Relaxed colour balance (CLRS, Ch. 13) |
| Splay Tree | Advanced | Self-adjusting: every access splays the node to the root |

All four share one interface: `insert(key)`, `search(key)`, `delete(key)`, `height()` and `len(tree)`. Each operation returns `True` or `False` to report success. Duplicate keys are ignored. Every algorithm is iterative, so degenerate trees (for example, a BST built from sorted input) cannot exceed Python's recursion limit. No library implementations of these data structures are used.

**Author:** Parinda Rahman
**Dataset:** https://ubcca-my.sharepoint.com/:f:/r/personal/parinda1_student_ubc_ca/Documents/COSC520%20Assignment%202%20Dataset?d=wf6bfeb25ac264f01b74bf6ae38692f08&csf=1&web=1&e=BP6fik

---

## Repository structure

```
cosc520-a2/
├── trees/
│   ├── bst.py             # Standard BST (baseline); Node class
│   ├── avl.py             # AVL tree (inherits from BST)
│   ├── red_black.py       # Red-Black tree (inherits from BST)
│   └── splay.py           # Splay tree (inherits from BST)
├── benchmarks/
│   ├── generate_data.py   # Synthetic dataset generator (.npz and CSV)
│   ├── run_benchmarks.py  # Benchmark harness; writes results/*.csv
│   └── plot_results.py    # Draws Figures 1-5 into results/figures/
├── tests/                 # pytest unit tests for every module
├── results/
│   ├── random.csv         # Raw measurements, main experiment (one row per run)
│   ├── workloads.csv      # Raw measurements, workload experiment
│   └── figures/           # fig1_build_time.png ... fig5_height.png
├── data/                  # Generated datasets (not in Git; see "Dataset")
├── pyproject.toml         # pytest configuration
└── requirements.txt       # Python dependencies
```

---

## Setup

Requires **Python 3.11 or newer** (developed and tested on Python 3.14).

```bash
git clone https://github.com/parindarahman/cosc520-a2.git
cd cosc520-a2
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate         # Windows (PowerShell)
source .venv/bin/activate      # macOS / Linux
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

The dependencies are `pytest` (tests), `numpy` (dataset generation), `pandas` (reading results) and `matplotlib` (plots). None of them implement tree structures.

On macOS/Linux you may need `python3` instead of `python`. On Windows, if PowerShell refuses to activate the environment, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

All commands below are run from the project root with the virtual environment active.

---

## Running the tests

```bash
pytest -v
```

The suite (85 tests) covers:

- **Every tree:** empty and single-node trees, duplicates, successful and unsuccessful searches, and deletion of leaves, one-child nodes, two-child nodes and the root. Each tree also gets 5,000 random operations checked against Python's built-in `set`.
- **AVL:** correct stored heights and every balance factor in {−1, 0, 1}; all four rotation cases; the height bound h < 1.44 log₂(n+2).
- **Red-Black:** black root, no red node with a red child, equal black-height on all paths, correct parent links; the height bound h ≤ 2 log₂(n+1).
- **Splay:** accessed nodes move to the root; the rotation counts for the zig, zig-zig and zig-zag steps.
- **Dataset generator:** distinct keys, reproducibility, query properties and CSV export.
- **Benchmark and plotting scripts:** end-to-end runs on tiny data.

To run a single file, pass its path, for example `pytest tests/test_avl.py -v`.

---

## Reproducing the experiments

### 1. Generate the datasets

```bash
python -m benchmarks.generate_data --csv
```

This writes `data/dataset_<n>.npz` and `data/csv/n_<n>/*.csv` for n = 10³, 10⁴, 10⁵, 10⁶ and 10⁷. It takes about 15 seconds. Omit `--csv` to write only the `.npz` files, which are what the benchmark reads. The benchmark also generates any missing dataset automatically.

### 2. Run the benchmarks

Quick trial (about 1 minute):

```bash
python -m benchmarks.run_benchmarks --sizes 1000 10000 100000 --runs 2 --workload-n 100000
```

Full run, as reported (about 30–60 minutes; needs roughly 3 GB of free RAM):

```bash
python -m benchmarks.run_benchmarks
```

The full run performs two experiments, each repeated 3 times:

- **Random experiment** (`results/random.csv`): for each n from 10³ to 10⁷, each tree is built from the shuffled keys. The script records build time, mean time per uniform search and per deletion, memory, height after building, and rotation count.
- **Workload experiment** (`results/workloads.csv`): at n = 10⁶, the script compares random, ascending and descending insertion orders, and uniform versus hot-key searches. The BST's sorted-input runs are capped at 10⁴ keys, because sorted input makes a plain BST quadratic.

Results are written row by row, so a partial run still leaves usable data. Timings depend on hardware; for stable results, keep the machine plugged in and close other heavy programs.

### 3. Draw the figures

```bash
python -m benchmarks.plot_results
```

This writes five PNG figures to `results/figures/`, each showing the mean of the runs:

| File | Content |
|---|---|
| `fig1_build_time.png` | Build time vs n (log-log) |
| `fig2_search_time.png` | Mean search time vs n (log-log) |
| `fig3_delete_time.png` | Mean deletion time vs n (log-log) |
| `fig4_memory.png` | Memory vs n (log-log) |
| `fig5_height.png` | Tree height vs n (log x-axis) |



## Dataset

For each n ∈ {10³, 10⁴, 10⁵, 10⁶, 10⁷}, the generator produces the following, with q = min(n, 10⁵):

| Array | Contents |
|---|---|
| `keys` | The n distinct even integers 0, 2, …, 2(n−1) in random order (the insertion order) |
| `uniform_queries` | q keys sampled uniformly with replacement (successful searches) |
| `miss_queries` | q odd integers, never present (unsuccessful searches) |
| `hot_keys` | A random 1% of the keys |
| `hot_queries` | q searches, 90% targeting `hot_keys` |
| `delete_keys` | q distinct keys to remove |


To use the hosted copy instead of regenerating, unzip the data file.

---

---

## Use of generative AI

In line with the course policy, generative AI (Anthropic's Claude) was used during this assignment. The implementation code, unit tests and benchmarking scripts were  ideated, verified with AI assistance, then reviewed, run and tested by the author.
---