"""Standard (unbalanced) binary search tree -- the baseline structure.

"""


class Node:
    """A single tree node holding a key and links to its two children."""

    __slots__ = ("key", "left", "right")  # saves memory at 10 million nodes

    def __init__(self, key):
        """
        Input: key -- a comparable value stored in this node.
        Output: None.
        Creates a leaf node with no children.
        """
        self.key = key
        self.left = None
        self.right = None


class BST:
    """Unbalanced binary search tree that stores distinct keys.

    Duplicate keys are ignored. All operations run in O(h) time, where
    h is the tree height: O(log n) on average for random input, but O(n)
    in the worst case (e.g., sorted input)..
    """

    def __init__(self):
        """
        Input: none.
        Output: None.
        Creates an empty tree.
        """
        self.root = None
        self.size = 0
        self.rotations = 0  # always 0 for BST; kept so all trees share one interface

    def __len__(self):
        """Output: the number of keys stored in the tree."""
        return self.size

    def search(self, key):
        """
        Input: key -- the value to look for.
        Output: True if key is in the tree, otherwise False.
        Walks down from the root, going left if key is smaller than the
        current node and right if larger.
        """
        node = self.root
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def insert(self, key):
        """
        Input: key -- the value to add.
        Output: True if the key was inserted, False if it was already present.
        Walks down from the root to the empty spot where the key belongs,
        then attaches a new leaf there.
        """
        if self.root is None:
            self.root = Node(key)
            self.size += 1
            return True

        node = self.root
        while True:
            if key == node.key:
                return False  # duplicate: tree unchanged
            if key < node.key:
                if node.left is None:
                    node.left = Node(key)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(key)
                    break
                node = node.right

        self.size += 1
        return True

    def delete(self, key):
        """
        Input: key -- the value to remove.
        Output: True if the key was removed, False if it was not found.
        Finds the node and removes it. A node with two children takes the
        key of its in-order successor (leftmost node of its right subtree),
        and the successor -- which has at most one child -- is removed instead.
        """
        parent, node = None, self.root
        while node is not None and node.key != key:
            parent = node
            node = node.left if key < node.key else node.right
        if node is None:
            return False  # key not in tree

        if node.left is not None and node.right is not None:
            succ_parent, succ = node, node.right
            while succ.left is not None:
                succ_parent, succ = succ, succ.left
            node.key = succ.key
            parent, node = succ_parent, succ  # now remove the successor

        child = node.left if node.left is not None else node.right
        self._replace_child(parent, node, child)
        self.size -= 1
        return True

    def _replace_child(self, parent, old, new):
        """
        Input: parent -- parent of `old` (None if `old` is the root);
               old -- the node being unlinked; new -- node (or None) to put in its place.
        Output: None.
        Points the correct link (root, parent.left or parent.right) at `new`.
        """
        if parent is None:
            self.root = new
        elif parent.left is old:
            parent.left = new
        else:
            parent.right = new

    def height(self):
        """
        Input: none.
        Output: the number of nodes on the longest root-to-leaf path
                (0 for an empty tree, 1 for a single node).
        Uses level-order traversal, counting levels.
        """
        if self.root is None:
            return 0
        level = [self.root]
        height = 0
        while level:
            height += 1
            level = [child for node in level
                     for child in (node.left, node.right) if child is not None]
        return height

    def inorder(self):
        """
        Input: none.
        Output: a list of all keys in ascending order.
        Iterative in-order traversal using an explicit stack; mainly used
        by unit tests to check that BST ordering holds.
        """
        keys, stack, node = [], [], self.root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            keys.append(node.key)
            node = node.right
        return keys