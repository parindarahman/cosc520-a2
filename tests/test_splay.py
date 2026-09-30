"""Unit tests for the Splay tree.

Checks normal insert/search/delete behaviour, that accessed nodes move to
the root, the rotation counts of the zig / zig-zig / zig-zag steps, and the
structural invariants (BST ordering, correct parent links, size) after
changes. Splay trees have no balance invariant to check.
"""

import random

from trees.splay import SplayTree


def assert_valid_splay(tree):
    """
    Input: tree -- a SplayTree.
    Output: None (raises AssertionError if the structure is invalid).
    Iteratively checks parent links, then checks in-order keys are strictly
    increasing and match the size counter. Iterative because splay trees
    can be very deep (e.g. height n after sorted inserts).
    """
    if tree.root is not None:
        assert tree.root.parent is None, "root has a parent"
    stack = [tree.root] if tree.root is not None else []
    while stack:
        node = stack.pop()
        for child in (node.left, node.right):
            if child is not None:
                assert child.parent is node, f"wrong parent link at key {child.key}"
                stack.append(child)
    keys = tree.inorder()
    assert all(a < b for a, b in zip(keys, keys[1:])), "BST ordering violated"
    assert len(keys) == len(tree), "size counter out of sync"


def build(keys):
    """Input: keys -- iterable of keys. Output: a SplayTree containing them."""
    tree = SplayTree()
    for key in keys:
        tree.insert(key)
    return tree


# ---------- empty and single-node trees ----------

def test_empty_tree():
    tree = SplayTree()
    assert len(tree) == 0
    assert tree.height() == 0
    assert not tree.search(1)
    assert not tree.delete(1)


def test_single_node():
    tree = build([10])
    assert tree.search(10)
    assert tree.root.key == 10
    assert tree.rotations == 0
    assert_valid_splay(tree)


# ---------- accesses move nodes to the root ----------

def test_insert_makes_new_key_root():
    tree = build([50, 30, 70, 20, 40])
    assert tree.insert(35)
    assert tree.root.key == 35
    assert_valid_splay(tree)


def test_search_hit_moves_key_to_root():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.search(20)
    assert tree.root.key == 20
    assert_valid_splay(tree)


def test_search_miss_splays_neighbour_to_root():
    tree = build([10, 20, 30, 40])
    assert not tree.search(25)
    assert tree.root.key in (20, 30)  # last node visited: predecessor or successor
    assert_valid_splay(tree)


def test_duplicate_insert_splays_existing_node():
    tree = build([5, 3, 8, 1])
    assert not tree.insert(8)
    assert tree.root.key == 8
    assert len(tree) == 4
    assert_valid_splay(tree)


def test_repeated_access_is_free():
    """A key accessed twice in a row is already at the root the second
    time, so no rotations are needed -- the adaptive behaviour."""
    tree = build(random.Random(3).sample(range(1000), 200))
    tree.search(tree.inorder()[0])
    rotations_before = tree.rotations
    assert tree.search(tree.root.key)
    assert tree.rotations == rotations_before


# ---------- the three splay steps ----------

def test_zig_step():
    tree = build([1])
    tree.insert(2)                # 2 is right child of root: one rotation
    assert tree.rotations == 1
    assert tree.root.key == 2 and tree.root.left.key == 1


def test_zig_zig_step():
    tree = build([1, 2, 3])       # left chain: 3 -> 2 -> 1
    assert tree.root.key == 3 and tree.root.left.left.key == 1
    rotations_before = tree.rotations
    tree.search(1)                # 1 and 2 are both left children
    assert tree.rotations - rotations_before == 2
    assert tree.root.key == 1
    assert tree.root.right.key == 2 and tree.root.right.right.key == 3
    assert_valid_splay(tree)


def test_zig_zag_step():
    tree = build([1, 3])          # 3 is root, 1 its left child
    rotations_before = tree.rotations
    tree.insert(2)                # 2 is right child of left child 1
    assert tree.rotations - rotations_before == 2
    assert tree.root.key == 2
    assert tree.root.left.key == 1 and tree.root.right.key == 3
    assert_valid_splay(tree)


# ---------- degenerate shapes ----------

def test_sorted_insert_builds_chain_without_recursion_error():
    """Sorted inserts leave a chain of height n (each insert is one cheap
    zig). n exceeds Python's recursion limit, so this also confirms the
    operations are iterative."""
    n = 5000
    tree = build(range(n))
    assert tree.height() == n
    assert_valid_splay(tree)


def test_deep_access_roughly_halves_height():
    """Splaying the deepest node of a chain folds the path, roughly
    halving the tree height -- why the amortized bound holds."""
    n = 1000
    tree = build(range(n))
    assert tree.search(0)
    assert tree.root.key == 0
    assert tree.height() <= n // 2 + 2
    assert_valid_splay(tree)


# ---------- deletion ----------

def test_delete_leaf():
    tree = build([50, 30, 70])
    assert tree.delete(30)
    assert not tree.search(30)
    assert_valid_splay(tree)


def test_delete_node_with_two_children():
    tree = build([50, 30, 70, 20, 40, 60, 80])
    assert tree.delete(50)
    assert tree.inorder() == [20, 30, 40, 60, 70, 80]
    assert tree.root.key == 40    # largest key smaller than 50 becomes root
    assert_valid_splay(tree)


def test_delete_smallest_key():
    tree = build([50, 30, 70, 20])
    assert tree.delete(20)        # splayed node has no left subtree
    assert tree.inorder() == [30, 50, 70]
    assert_valid_splay(tree)


def test_delete_only_node():
    tree = build([7])
    assert tree.delete(7)
    assert tree.root is None
    assert len(tree) == 0


def test_delete_missing_key_splays_neighbour():
    tree = build([10, 20, 30])
    assert not tree.delete(25)
    assert tree.root.key in (20, 30)
    assert len(tree) == 3
    assert_valid_splay(tree)


def test_delete_everything():
    keys = list(range(500))
    tree = build(keys)
    random.Random(7).shuffle(keys)
    for key in keys:
        assert tree.delete(key)
        assert_valid_splay(tree)
    assert tree.root is None
    assert len(tree) == 0


# ---------- randomized check against a reference model ----------

def test_random_operations_match_set():
    """Applies 5000 random inserts/deletes/searches, checks every result
    against Python's built-in set, and checks the structure throughout."""
    rng = random.Random(42)
    tree, reference = SplayTree(), set()
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
            assert_valid_splay(tree)
    assert tree.inorder() == sorted(reference)
    assert_valid_splay(tree)