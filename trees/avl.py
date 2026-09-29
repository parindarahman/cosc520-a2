"""AVL tree: a height-balanced binary search tree.



AVLTree reuses search(), inorder(), __len__() and _replace_child() from BST.
All operations are iterative, so they cannot hit Python's recursion limit.
"""

from trees.bst import BST, Node


class AVLNode(Node):
    """A BST node that also stores the height of its subtree."""

    __slots__ = ("height",)

    def __init__(self, key):
        """
        Input: key -- a comparable value stored in this node.
        Output: None.
        Creates a leaf node; a leaf has height 1.
        """
        super().__init__(key)
        self.height = 1


def _height(node):
    """
    Input: node -- an AVLNode or None.
    Output: the stored subtree height, or 0 for an empty subtree.
    """
    return node.height if node is not None else 0


def _update_height(node):
    """
    Input: node -- an AVLNode whose children have correct heights.
    Output: None.
    Recomputes node.height from its children's heights.
    """
    node.height = 1 + max(_height(node.left), _height(node.right))


def _balance_factor(node):
    """
    Input: node -- an AVLNode.
    Output: height(left subtree) - height(right subtree).
    """
    return _height(node.left) - _height(node.right)


class AVLTree(BST):
    """AVL tree storing distinct keys, with the same interface as BST."""

    # ---------- rotations ----------

    def _rotate_right(self, top):
        """
        Input: top -- a node whose left child becomes the new subtree root.
        Output: the new root of this subtree.

                top              new_top
               /    \\            /     \\
           new_top   C    ->    A      top
            /   \\                     /   \\
           A     B                   B     C
        """
        new_top = top.left
        top.left = new_top.right
        new_top.right = top
        _update_height(top)       # top is now lower, so update it first
        _update_height(new_top)
        self.rotations += 1
        return new_top

    def _rotate_left(self, top):
        """
        Input: top -- a node whose right child becomes the new subtree root.
        Output: the new root of this subtree (mirror image of _rotate_right).
        """
        new_top = top.right
        top.right = new_top.left
        new_top.left = top
        _update_height(top)
        _update_height(new_top)
        self.rotations += 1
        return new_top

    def _rebalance(self, node):
        """
        Input: node -- a node whose children are valid AVL subtrees.
        Output: the root of this subtree after any needed rotations.
        Updates node's height, then fixes the four imbalance cases:
        left-left and right-right need one rotation; left-right and
        right-left need two.
        """
        _update_height(node)
        balance = _balance_factor(node)

        if balance > 1:                                  # left side too tall
            if _balance_factor(node.left) < 0:           # left-right case
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)              # left-left case

        if balance < -1:                                 # right side too tall
            if _balance_factor(node.right) > 0:          # right-left case
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)               # right-right case

        return node

    def _rebalance_path(self, path):
        """
        Input: path -- list of nodes from the root down to the parent of
                       the position that was just changed.
        Output: None.
        Walks back up the path, rebalancing each node and reattaching the
        result to its parent. Stops early once a subtree's root and height
        are both unchanged, because nothing above it can be affected.
        """
        for depth in range(len(path) - 1, -1, -1):
            node = path[depth]
            old_height = node.height
            new_subtree = self._rebalance(node)

            parent = path[depth - 1] if depth > 0 else None
            self._replace_child(parent, node, new_subtree)

            if new_subtree is node and node.height == old_height:
                break

    # ---------- public operations ----------

    def insert(self, key):
        """
        Input: key -- the value to add.
        Output: True if the key was inserted, False if it was already present.
        Inserts a leaf as in a plain BST, then rebalances on the way back up.
        At most one single or double rotation is needed per insertion.
        """
        if self.root is None:
            self.root = AVLNode(key)
            self.size += 1
            return True

        path, node = [], self.root
        while node is not None:
            if key == node.key:
                return False  # duplicate: tree unchanged
            path.append(node)
            node = node.left if key < node.key else node.right

        parent = path[-1]
        if key < parent.key:
            parent.left = AVLNode(key)
        else:
            parent.right = AVLNode(key)

        self.size += 1
        self._rebalance_path(path)
        return True

    def delete(self, key):
        """
        Input: key -- the value to remove.
        Output: True if the key was removed, False if it was not found.
        Removes the node as in a plain BST (a node with two children takes
        its in-order successor's key, and the successor is removed instead),
        then rebalances on the way back up. Unlike insertion, a deletion may
        need rotations at several levels, O(log n) in the worst case.
        """
        path, node = [], self.root
        while node is not None and node.key != key:
            path.append(node)
            node = node.left if key < node.key else node.right
        if node is None:
            return False  # key not in tree

        if node.left is not None and node.right is not None:
            path.append(node)
            successor = node.right
            while successor.left is not None:
                path.append(successor)
                successor = successor.left
            node.key = successor.key
            node = successor  # now remove the successor (at most one child)

        child = node.left if node.left is not None else node.right
        parent = path[-1] if path else None
        self._replace_child(parent, node, child)

        self.size -= 1
        self._rebalance_path(path)
        return True

    def height(self):
        """
        Input: none.
        Output: the number of nodes on the longest root-to-leaf path
                (0 for an empty tree), same convention as BST.height().
        O(1): the root already stores its subtree height.
        """
        return _height(self.root)