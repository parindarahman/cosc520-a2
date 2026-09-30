"""Splay tree: a self-adjusting binary search tree.

A splay tree stores no balance information at all. Instead, every access
(search, insert or delete) moves the accessed node to the root with a
sequence of rotations called "splaying". Recently and frequently accessed
keys therefore stay near the top, which makes splay trees fast on skewed
workloads (a few "hot" keys). Individual operations can cost O(n), but any
sequence of m operations costs O(m log n) in total, i.e. O(log n) amortized
per operation (Sleator & Tarjan, "Self-Adjusting Binary Search Trees",
JACM 32(3), 1985).

This is the bottom-up variant with parent pointers. Splaying repeats three
steps until the node reaches the root:
    zig      -- parent is the root: one rotation;
    zig-zig  -- node and parent are both left (or both right) children:
                rotate the parent first, then the node;
    zig-zag  -- one is a left child and the other a right child:
                rotate the node twice.

SplayTree reuses inorder(), height() and __len__() from BST. search() is
overridden because, unlike in the other trees, searching changes the tree.

"""

from trees.bst import BST, Node


class SplayNode(Node):
    """A BST node that also stores a link to its parent."""

    __slots__ = ("parent",)

    def __init__(self, key, parent=None):
        """
        Input: key -- a comparable value; parent -- parent node (None for root).
        Output: None.
        """
        super().__init__(key)
        self.parent = parent


class SplayTree(BST):
    """Splay tree storing distinct keys, with the same interface as BST."""

    # ---------- splaying ----------

    def _rotate_up(self, node):
        """
        Input: node -- a non-root node.
        Output: None.
        Rotates node above its parent (a right rotation if node is a left
        child, a left rotation if it is a right child), fixing all parent
        links and the grandparent's (or root's) link.
        """
        parent = node.parent
        grandparent = parent.parent

        if node is parent.left:
            parent.left = node.right
            if node.right is not None:
                node.right.parent = parent
            node.right = parent
        else:
            parent.right = node.left
            if node.left is not None:
                node.left.parent = parent
            node.left = parent

        parent.parent = node
        node.parent = grandparent
        if grandparent is None:
            self.root = node
        elif grandparent.left is parent:
            grandparent.left = node
        else:
            grandparent.right = node
        self.rotations += 1

    def _splay(self, node):
        """
        Input: node -- a node in this tree.
        Output: None.
        Moves node to the root using zig, zig-zig and zig-zag steps.
        Zig-zig rotates the parent before the node; this is what roughly
        halves the depth of nodes along the access path and gives the
        O(log n) amortized bound.
        """
        while node.parent is not None:
            parent = node.parent
            grandparent = parent.parent
            if grandparent is None:                                  # zig
                self._rotate_up(node)
            elif (node is parent.left) == (parent is grandparent.left):
                self._rotate_up(parent)                              # zig-zig
                self._rotate_up(node)
            else:                                                    # zig-zag
                self._rotate_up(node)
                self._rotate_up(node)

    def _find(self, key):
        """
        Input: key -- the value to look for.
        Output: (node, last) where node holds key (or None if absent) and
                last is the last node visited (None only for an empty tree).
        Plain BST descent; does not change the tree.
        """
        node, last = self.root, None
        while node is not None:
            last = node
            if key == node.key:
                return node, last
            node = node.left if key < node.key else node.right
        return None, last

    # ---------- public operations ----------

    def search(self, key):
        """
        Input: key -- the value to look for.
        Output: True if key is in the tree, otherwise False.
        Splays the found node to the root. On a miss, splays the last node
        visited instead, so unsuccessful searches are also paid for by
        restructuring (required for the amortized bound).
        """
        node, last = self._find(key)
        if last is not None:
            self._splay(node if node is not None else last)
        return node is not None

    def insert(self, key):
        """
        Input: key -- the value to add.
        Output: True if the key was inserted, False if it was already present.
        Inserts a leaf as in a plain BST and splays it to the root.
        A duplicate is not inserted, but its existing node is still splayed.
        """
        node, last = self._find(key)
        if node is not None:
            self._splay(node)
            return False

        new_node = SplayNode(key, last)
        if last is None:
            self.root = new_node
        elif key < last.key:
            last.left = new_node
        else:
            last.right = new_node

        self.size += 1
        self._splay(new_node)
        return True

    def delete(self, key):
        """
        Input: key -- the value to remove.
        Output: True if the key was removed, False if it was not found.
        Splays the node to the root and removes it, leaving two subtrees
        L (smaller keys) and R (larger keys). The largest key in L is
        splayed to the top of L; it then has no right child, so R is
        attached there. If the key is missing, the last node visited is
        splayed instead.
        """
        node, last = self._find(key)
        if node is None:
            if last is not None:
                self._splay(last)
            return False

        self._splay(node)
        left, right = node.left, node.right

        if left is None:
            self.root = right
            if right is not None:
                right.parent = None
        else:
            left.parent = None
            self.root = left              # splay within L as its own tree
            largest = left
            while largest.right is not None:
                largest = largest.right
            self._splay(largest)          # largest is now root of L
            largest.right = right
            if right is not None:
                right.parent = largest

        self.size -= 1
        return True