"""End-to-end tests for the benchmark and plotting scripts.

Runs each experiment on tiny datasets in a temporary folder and checks the
CSVs and figures are produced correctly.
"""

import pandas as pd
import pytest

from benchmarks import plot_results
from benchmarks.run_benchmarks import (
    STRUCTURES, build_tree, run_random_experiment, run_workload_experiment,
    time_per_operation,
)

SIZES = [200, 500]
RUNS = 2


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """
    Output: a temporary folder in which both experiments have been run on
    tiny data. Runs once for all tests in this file, since it is the slow part.
    """
    folder = tmp_path_factory.mktemp("results")
    data_dir = str(folder / "data")
    run_random_experiment(SIZES, RUNS, str(folder / "random.csv"), data_dir)
    run_workload_experiment(500, RUNS, 100, str(folder / "workloads.csv"), data_dir)
    return folder


@pytest.fixture
def point_scripts_at(results, monkeypatch):
    """Output: results folder, with the plotting script redirected to it."""
    monkeypatch.setattr(plot_results, "RESULTS_DIR", str(results))
    monkeypatch.setattr(plot_results, "FIGURES_DIR", str(results / "figures"))
    return results


def test_build_tree_inserts_every_key():
    for tree_class in STRUCTURES.values():
        tree, seconds = build_tree(tree_class, [5, 3, 8, 1])
        assert len(tree) == 4 and seconds >= 0


def test_time_per_operation_handles_empty_input():
    assert time_per_operation(print, []) == 0.0


def test_random_csv_complete(results):
    df = pd.read_csv(results / "random.csv")
    assert len(df) == len(SIZES) * len(STRUCTURES) * RUNS
    assert (df["height"] > 0).all()
    assert (df["memory_mb"] > 0).all()
    assert (df[["build_s", "search_us", "delete_us"]] > 0).all().all()


def test_workload_csv_caps_bst_sorted_runs(results):
    df = pd.read_csv(results / "workloads.csv")
    sorted_rows = df["workload"].str.startswith("Sorted")
    bst = df["structure"] == "BST"
    assert (df.loc[sorted_rows & bst, "n_used"] == 100).all()
    assert (df.loc[~(sorted_rows & bst), "n_used"] == 500).all()
    assert set(df["workload"]) == {"Random", "Sorted ascending",
                                   "Sorted descending", "Hot-key search"}


def test_sorted_bst_is_degenerate(results):
    df = pd.read_csv(results / "workloads.csv")
    row = df[(df["workload"] == "Sorted ascending") & (df["structure"] == "BST")]
    assert (row["height"] == row["n_used"]).all()


def test_figures_written(point_scripts_at):
    results = point_scripts_at
    plot_results.main()
    for file_name, *_ in plot_results.FIGURES:
        assert (results / "figures" / f"{file_name}.png").exists()
        assert not (results / "figures" / f"{file_name}.pdf").exists()