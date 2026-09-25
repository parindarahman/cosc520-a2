"""Unit tests for the dataset generator.
"""

import numpy as np
import pytest

from benchmarks.generate_data import (
    HOT_KEY_FRACTION, MAX_QUERIES, csv_folder, generate_dataset, load_dataset,
    save_dataset,
)

N = 5000


@pytest.fixture(scope="module")
def dataset():
    """Output: a dataset of size N, generated once for all tests here."""
    return generate_dataset(N)


def test_keys_are_distinct_and_complete(dataset):
    keys = dataset["keys"]
    assert len(keys) == N
    assert len(set(keys.tolist())) == N


def test_keys_are_shuffled(dataset):
    keys = dataset["keys"]
    assert not np.array_equal(keys, np.sort(keys))


def test_query_counts(dataset):
    expected = min(N, MAX_QUERIES)
    for name in ["uniform_queries", "miss_queries", "hot_queries", "delete_keys"]:
        assert len(dataset[name]) == expected


def test_uniform_queries_all_hit(dataset):
    assert np.isin(dataset["uniform_queries"], dataset["keys"]).all()


def test_miss_queries_never_hit(dataset):
    assert not np.isin(dataset["miss_queries"], dataset["keys"]).any()


def test_hot_queries_are_skewed(dataset):
    hot_keys = dataset["keys"][:max(1, int(N * HOT_KEY_FRACTION))]
    share_hot = np.isin(dataset["hot_queries"], hot_keys).mean()
    assert share_hot > 0.85  # ~0.90 expected; allows for sampling noise


def test_delete_keys_distinct_and_present(dataset):
    deletes = dataset["delete_keys"]
    assert len(set(deletes.tolist())) == len(deletes)
    assert np.isin(deletes, dataset["keys"]).all()


def test_generation_is_reproducible():
    first, second = generate_dataset(1000), generate_dataset(1000)
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_invalid_size_rejected():
    with pytest.raises(ValueError):
        generate_dataset(0)


def test_save_and_load_round_trip(tmp_path):
    save_dataset(1000, data_dir=tmp_path)
    loaded = load_dataset(1000, data_dir=tmp_path)
    original = generate_dataset(1000)
    assert loaded["keys"] == original["keys"].tolist()
    assert isinstance(loaded["keys"][0], int)  # plain Python ints, not numpy


def test_csv_export_matches_dataset(tmp_path):
    save_dataset(1000, data_dir=tmp_path, write_csv=True)
    original = generate_dataset(1000)
    folder = csv_folder(1000, data_dir=tmp_path)
    for name, values in original.items():
        path = f"{folder}/{name}.csv"
        with open(path) as f:
            assert f.readline().strip() == name  # header row
        from_csv = np.loadtxt(path, dtype=np.int64, skiprows=1, ndmin=1)
        assert np.array_equal(from_csv, values)


def test_csv_not_written_by_default(tmp_path):
    save_dataset(1000, data_dir=tmp_path)
    assert not (tmp_path / "csv").exists()