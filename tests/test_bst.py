"""Unit tests for the baseline BST.

Authorship: generated with AI assistance (Claude); reviewed by Parinda Rahman.
"""

import random

import pytest

from trees.bst import BST


def assert_valid_bst(tree):
    """
    Input: tree -- a BST instance.
    Output: None (raises AssertionError if the tree is invalid).
    Checks that in-order keys are strictly increasing (BST ordering,
    no duplicates) and that the stored size matches the real node count.
    """
    keys = tree.inorder()
    assert all(a < b for a, b in zip(keys, keys[1:])), "BST ordering violated"
    assert len(keys) == len(tree), "size counter out of sync"


@pytest.fixture
def sample_tree():
    """Output: a BST built from a fixed key sequence with a known shape:

            50
          /    \\
        30      70
       /  \\    /  \\
      20  40  60   80
                     \\
                     90
    """
    tree = BST()
    for key in [50, 30, 70, 20, 40, 60, 80, 90]:
        tree.insert(key)
    return tree


# ---------- empty and single-node trees ----------

def test_empty_tree():
    tree = BST()
    assert len(tree) == 0
    assert tree.height() == 0
    assert tree.inorder() == []
    assert not tree.search(5)
    assert not tree.delete(5)


def test_single_node():
    tree = BST()
    assert tree.insert(10)
    assert tree.search(10)
    assert len(tree) == 1
    assert tree.height() == 1
    assert tree.root.key == 10


# ---------- insertion and search ----------

def test_insert_keeps_order(sample_tree):
    assert sample_tree.inorder() == [20, 30, 40, 50, 60, 70, 80, 90]
    assert_valid_bst(sample_tree)


def test_duplicate_insert_ignored(sample_tree):
    assert not sample_tree.insert(40)
    assert len(sample_tree) == 8
    assert_valid_bst(sample_tree)


def test_search_hit_and_miss(sample_tree):
    for key in [20, 50, 90]:
        assert sample_tree.search(key)
    for key in [0, 45, 100]:
        assert not sample_tree.search(key)


def test_height(sample_tree):
    assert sample_tree.height() == 4  # 50 -> 70 -> 80 -> 90


# ---------- deletion ----------

def test_delete_leaf(sample_tree):
    assert sample_tree.delete(20)
    assert not sample_tree.search(20)
    assert len(sample_tree) == 7
    assert_valid_bst(sample_tree)


def test_delete_node_with_one_child(sample_tree):
    assert sample_tree.delete(80)          # 80 has only right child 90
    assert sample_tree.root.right.right.key == 90
    assert_valid_bst(sample_tree)


def test_delete_node_with_two_children(sample_tree):
    assert sample_tree.delete(30)          # successor is 40
    assert sample_tree.root.left.key == 40
    assert sample_tree.inorder() == [20, 40, 50, 60, 70, 80, 90]
    assert_valid_bst(sample_tree)


def test_delete_root_with_two_children(sample_tree):
    assert sample_tree.delete(50)          # successor is 60
    assert sample_tree.root.key == 60
    assert not sample_tree.search(50)
    assert_valid_bst(sample_tree)


def test_delete_root_with_one_child():
    tree = BST()
    tree.insert(1)
    tree.insert(2)
    assert tree.delete(1)
    assert tree.root.key == 2
    assert_valid_bst(tree)


def test_delete_only_node():
    tree = BST()
    tree.insert(7)
    assert tree.delete(7)
    assert tree.root is None
    assert len(tree) == 0


def test_delete_missing_key(sample_tree):
    assert not sample_tree.delete(999)
    assert len(sample_tree) == 8


# ---------- degenerate and randomized behaviour ----------

def test_sorted_insert_degenerates_without_recursion_error():
    """Sorted input produces a linked-list-shaped tree of height n.
    n exceeds Python's default recursion limit (1000), so this also
    confirms every operation is iterative."""
    n = 5000
    tree = BST()
    for key in range(n):
        tree.insert(key)
    assert tree.height() == n
    assert tree.search(n - 1)
    assert tree.delete(0)
    assert_valid_bst(tree)


def test_random_operations_match_set():
    """Applies 5000 random inserts/deletes/searches and checks every
    result against Python's built-in set as a reference model."""
    rng = random.Random(42)
    tree, reference = BST(), set()
    for _ in range(5000):
        key = rng.randint(0, 500)
        action = rng.choice(["insert", "delete", "search"])
        if action == "insert":
            assert tree.insert(key) == (key not in reference)
            reference.add(key)
        elif action == "delete":
            assert tree.delete(key) == (key in reference)
            reference.discard(key)
        else:
            assert tree.search(key) == (key in reference)
    assert tree.inorder() == sorted(reference)
    assert_valid_bst(tree)