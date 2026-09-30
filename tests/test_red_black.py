"""Unit tests for the Red-Black tree.

Besides normal insert/search/delete behaviour, these tests check every
Red-Black property after changes: black root, no red node with a red child,
equal black-height on all paths, BST ordering, and correct parent links.
"""

import math
import random

from trees.red_black import BLACK, RED, RedBlackTree


def check_subtree(node, parent=None, low=None, high=None):
    """
    Input: node -- subtree root (or None); parent -- expected parent of node;
           low/high -- exclusive key bounds.
    Output: the black-height of the subtree (empty subtree = 1).
    Raises AssertionError if ordering, parent links, colours or
    black-heights are wrong. Recursion is safe: RB trees are shallow.
    """
    if node is None:
        return 1
    assert node.parent is parent, f"wrong parent link at key {node.key}"
    assert low is None or node.key > low, "BST ordering violated"
    assert high is None or node.key < high, "BST ordering violated"
    if node.color == RED:
        assert node.left is None or node.left.color == BLACK, \
            f"red node {node.key} has a red child"
        assert node.right is None or node.right.color == BLACK, \
            f"red node {node.key} has a red child"
    left_black = check_subtree(node.left, node, low, node.key)
    right_black = check_subtree(node.right, node, node.key, high)
    assert left_black == right_black, f"black-heights differ at key {node.key}"
    return left_black + (1 if node.color == BLACK else 0)


def assert_valid_rb(tree):
    """
    Input: tree -- a RedBlackTree.
    Output: None (raises AssertionError if any Red-Black property fails).
    """
    if tree.root is not None:
        assert tree.root.color == BLACK, "root is not black"
    check_subtree(tree.root)
    assert len(tree.inorder()) == len(tree), "size counter out of sync"


def build(keys):
    """Input: keys -- iterable of keys. Output: a RedBlackTree containing them."""
    tree = RedBlackTree()
    for key in keys:
        tree.insert(key)
    return tree


# ---------- empty and single-node trees ----------

def test_empty_tree():
    tree = RedBlackTree()
    assert len(tree) == 0
    assert tree.height() == 0
    assert not tree.search(1)
    assert not tree.delete(1)


def test_single_node_is_black_root():
    tree = build([10])
    assert tree.search(10)
    assert tree.root.color == BLACK
    assert tree.root.parent is None
    assert_valid_rb(tree)


# ---------- insertion fix-up cases ----------

def test_insert_red_uncle_recolours_without_rotation():
    # Inserting 1 under red 5 with red uncle 15: recolour only.
    tree = build([10, 5, 15, 1])
    assert tree.rotations == 0
    assert tree.root.key == 10
    assert tree.root.left.color == BLACK and tree.root.right.color == BLACK
    assert tree.root.left.left.color == RED
    assert_valid_rb(tree)


def test_insert_straight_line_needs_one_rotation():
    tree = build([1, 2, 3])
    assert tree.rotations == 1
    assert tree.root.key == 2 and tree.root.color == BLACK
    assert tree.root.left.color == RED and tree.root.right.color == RED
    assert_valid_rb(tree)


def test_insert_zigzag_needs_two_rotations():
    tree = build([3, 1, 2])
    assert tree.rotations == 2
    assert tree.root.key == 2
    assert_valid_rb(tree)


def test_duplicate_insert_ignored():
    tree = build([5, 3, 8])
    assert not tree.insert(3)
    assert len(tree) == 3
    assert_valid_rb(tree)


def test_search_hit_and_miss():
    tree = build(range(0, 100, 2))
    assert all(tree.search(k) for k in range(0, 100, 2))
    assert not any(tree.search(k) for k in range(1, 100, 2))


# ---------- balance guarantee ----------

def test_sorted_insert_stays_balanced():
    """Sorted input degenerates a plain BST to height n; a Red-Black tree
    must stay within the bound h <= 2 * log2(n + 1)."""
    n = 10_000
    tree = build(range(n))
    assert tree.height() <= 2 * math.log2(n + 1)
    assert_valid_rb(tree)


# ---------- deletion ----------

def test_delete_red_leaf_needs_no_fixup():
    tree = build([10, 5, 15, 1])      # 1 is a red leaf
    rotations_before = tree.rotations
    assert tree.delete(1)
    assert tree.rotations == rotations_before
    assert_valid_rb(tree)


def test_delete_black_leaf_triggers_rotation():
    #   10(B)                      15(B)
    #   /   \\                     /    \\
    # 5(B)  15(B)       ->     10(B)   20(B)     (after deleting 5)
    #          \\
    #          20(R)
    tree = build([10, 5, 15, 20])
    rotations_before = tree.rotations
    assert tree.delete(5)
    assert tree.root.key == 15
    assert tree.rotations == rotations_before + 1
    assert_valid_rb(tree)


def test_delete_node_with_two_children():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.delete(30)            # successor is 40
    assert tree.inorder() == [20, 40, 50, 60, 70, 80]
    assert_valid_rb(tree)


def test_delete_root():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.delete(50)
    assert not tree.search(50)
    assert tree.root.parent is None
    assert_valid_rb(tree)


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
        assert_valid_rb(tree)
    assert tree.root is None
    assert len(tree) == 0


# ---------- randomized check against a reference model ----------

def test_random_operations_match_set():
    """Applies 5000 random inserts/deletes/searches, checks every result
    against Python's built-in set, and checks the RB properties throughout."""
    rng = random.Random(42)
    tree, reference = RedBlackTree(), set()
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
            assert_valid_rb(tree)
    assert tree.inorder() == sorted(reference)
    assert_valid_rb(tree)