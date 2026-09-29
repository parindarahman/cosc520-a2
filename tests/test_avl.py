"""Unit tests for the AVL tree.

Besides normal insert/search/delete behaviour, these tests check the AVL
invariants after every change: BST ordering, correct stored heights, and
every balance factor in {-1, 0, 1}.
"""

import math
import random

import pytest

from trees.avl import AVLTree
from trees.bst import BST


def check_subtree(node, low=None, high=None):
    """
    Input: node -- subtree root (or None); low/high -- exclusive key bounds.
    Output: the true height of the subtree, computed from scratch.
    Raises AssertionError if ordering, stored height or balance is wrong.
    Recursion is safe here because AVL trees are shallow.
    """
    if node is None:
        return 0
    assert low is None or node.key > low, "BST ordering violated"
    assert high is None or node.key < high, "BST ordering violated"
    left_height = check_subtree(node.left, low, node.key)
    right_height = check_subtree(node.right, node.key, high)
    true_height = 1 + max(left_height, right_height)
    assert node.height == true_height, f"stale height at key {node.key}"
    assert abs(left_height - right_height) <= 1, f"unbalanced at key {node.key}"
    return true_height


def assert_valid_avl(tree):
    """
    Input: tree -- an AVLTree.
    Output: None (raises AssertionError if any AVL property fails).
    """
    check_subtree(tree.root)
    assert len(tree.inorder()) == len(tree), "size counter out of sync"


def build(keys):
    """Input: keys -- iterable of keys. Output: an AVLTree containing them."""
    tree = AVLTree()
    for key in keys:
        tree.insert(key)
    return tree


# ---------- empty and single-node trees ----------

def test_empty_tree():
    tree = AVLTree()
    assert len(tree) == 0
    assert tree.height() == 0
    assert not tree.search(1)
    assert not tree.delete(1)


def test_single_node():
    tree = build([10])
    assert tree.search(10)
    assert tree.height() == 1
    assert tree.rotations == 0
    assert_valid_avl(tree)


# ---------- the four rotation cases on insert ----------

@pytest.mark.parametrize("keys, expected_rotations", [
    ([3, 2, 1], 1),   # left-left: single right rotation
    ([1, 2, 3], 1),   # right-right: single left rotation
    ([3, 1, 2], 2),   # left-right: double rotation
    ([1, 3, 2], 2),   # right-left: double rotation
])
def test_insert_rotation_cases(keys, expected_rotations):
    tree = build(keys)
    assert tree.root.key == 2
    assert tree.root.left.key == 1 and tree.root.right.key == 3
    assert tree.rotations == expected_rotations
    assert_valid_avl(tree)


def test_duplicate_insert_ignored():
    tree = build([5, 3, 8])
    assert not tree.insert(3)
    assert len(tree) == 3
    assert_valid_avl(tree)


def test_search_hit_and_miss():
    tree = build(range(0, 100, 2))
    assert all(tree.search(k) for k in range(0, 100, 2))
    assert not any(tree.search(k) for k in range(1, 100, 2))


# ---------- balance guarantees ----------

def test_sorted_insert_stays_balanced():
    """Sorted input degenerates a plain BST to height n; AVL must stay
    within the theoretical bound h < 1.44 * log2(n + 2)."""
    n = 10_000
    tree = build(range(n))
    assert tree.height() < 1.44 * math.log2(n + 2)
    assert_valid_avl(tree)


def test_stored_height_matches_level_count():
    """AVLTree.height() is O(1) from the root; it must agree with BST's
    level-by-level count on the same tree."""
    tree = build(random.Random(1).sample(range(10_000), 2_000))
    assert tree.height() == BST.height(tree)


# ---------- deletion ----------

def test_delete_leaf():
    tree = build([2, 1, 3])
    assert tree.delete(1)
    assert not tree.search(1)
    assert_valid_avl(tree)


def test_delete_triggers_rotation():
    #     2              3
    #    / \            / \
    #   1   3    ->    2   4     (after deleting 1)
    #        \
    #         4
    tree = build([2, 1, 3, 4])
    rotations_before = tree.rotations
    assert tree.delete(1)
    assert tree.root.key == 3
    assert tree.rotations > rotations_before
    assert_valid_avl(tree)


def test_delete_node_with_two_children():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.delete(30)            # successor is 40
    assert tree.inorder() == [20, 40, 50, 60, 70, 80]
    assert_valid_avl(tree)


def test_delete_root():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.delete(50)
    assert not tree.search(50)
    assert tree.root.key == 60
    assert_valid_avl(tree)


def test_delete_missing_key():
    tree = build([1, 2, 3])
    assert not tree.delete(99)
    assert len(tree) == 3


def test_delete_everything():
    keys = list(range(500))
    tree = build(keys)
    random.Random(7).shuffle(keys)
    for key in keys:
        assert tree.delete(key)
        assert_valid_avl(tree)
    assert tree.root is None
    assert len(tree) == 0


# ---------- randomized check against a reference model ----------

def test_random_operations_match_set():
    """Applies 5000 random inserts/deletes/searches, checks every result
    against Python's built-in set, and checks the AVL invariants throughout."""
    rng = random.Random(42)
    tree, reference = AVLTree(), set()
    for step in range(5000):
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
        if step % 100 == 0:
            assert_valid_avl(tree)
    assert tree.inorder() == sorted(reference)
    assert_valid_avl(tree)