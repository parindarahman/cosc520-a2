"""Red-Black tree: a colour-balanced binary search tree.

Every node is red or black, and the tree keeps these properties:
    1. The root is black.
    2. Empty subtrees (None) count as black.
    3. A red node never has a red child.
    4. Every path from a node down to an empty subtree passes through
       the same number of black nodes (the node's "black-height").
Together they guarantee height <= 2 * log2(n + 1), so search, insert and
delete are O(log n) worst case. Balance is looser than AVL (taller trees),
but fixing it needs fewer rotations: at most 2 per insert, 3 per delete.

RedBlackTree reuses search(), inorder(), height() and __len__() from BST.
"""

from trees.bst import BST, Node

RED = True
BLACK = False


class RBNode(Node):
    """A BST node that also stores its colour and a link to its parent."""

    __slots__ = ("color", "parent")

    def __init__(self, key, parent=None):
        """
        Input: key -- a comparable value; parent -- parent node (None for root).
        Output: None.
        Creates a red leaf: new nodes are always inserted red.
        """
        super().__init__(key)
        self.color = RED
        self.parent = parent


def _is_red(node):
    """
    Input: node -- an RBNode or None.
    Output: True if node is red; empty subtrees (None) count as black.
    """
    return node is not None and node.color == RED


class RedBlackTree(BST):
    """Red-Black tree storing distinct keys, with the same interface as BST."""

    # ---------- structural helpers ----------

    def _rotate_left(self, top):
        """
        Input: top -- a node whose right child becomes the new subtree root.
        Output: None (the tree is modified in place, parent links included).

              top                 new_top
             /   \\               /     \\
            A   new_top   ->    top      C
                /    \\         /   \\
               B      C        A     B
        """
        new_top = top.right
        top.right = new_top.left
        if new_top.left is not None:
            new_top.left.parent = top
        self._transplant(top, new_top)
        new_top.left = top
        top.parent = new_top
        self.rotations += 1

    def _rotate_right(self, top):
        """
        Input: top -- a node whose left child becomes the new subtree root.
        Output: None (mirror image of _rotate_left).
        """
        new_top = top.left
        top.left = new_top.right
        if new_top.right is not None:
            new_top.right.parent = top
        self._transplant(top, new_top)
        new_top.right = top
        top.parent = new_top
        self.rotations += 1

    def _transplant(self, old, new):
        """
        Input: old -- a node in the tree; new -- a node or None.
        Output: None.
        Puts `new` where `old` was, relative to old's parent (or as the root),
        and updates new's parent link. old's own children are left untouched.
        """
        parent = old.parent
        if parent is None:
            self.root = new
        elif parent.left is old:
            parent.left = new
        else:
            parent.right = new
        if new is not None:
            new.parent = parent

    # ---------- insertion ----------

    def insert(self, key):
        """
        Input: key -- the value to add.
        Output: True if the key was inserted, False if it was already present.
        Inserts a red leaf as in a plain BST, then restores the Red-Black
        properties with recolouring and at most two rotations.
        """
        parent, node = None, self.root
        while node is not None:
            if key == node.key:
                return False  # duplicate: tree unchanged
            parent = node
            node = node.left if key < node.key else node.right

        new_node = RBNode(key, parent)
        if parent is None:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        self.size += 1
        self._insert_fixup(new_node)
        return True

    def _insert_fixup(self, node):
        """
        Input: node -- a newly inserted red node.
        Output: None.
        While node and its parent are both red (property 3 broken):
          - red uncle:   recolour parent, uncle and grandparent, then move
                         the problem up to the grandparent;
          - black uncle: rotate once or twice at the parent/grandparent and
                         recolour, which ends the loop.
        Finally colours the root black (property 1).
        """
        while _is_red(node.parent):
            parent = node.parent
            grandparent = parent.parent  # exists: a red parent is never the root

            if parent is grandparent.left:
                uncle = grandparent.right
                if _is_red(uncle):                        # case 1: recolour
                    parent.color = uncle.color = BLACK
                    grandparent.color = RED
                    node = grandparent
                else:
                    if node is parent.right:              # case 2: zig-zag
                        node = parent
                        self._rotate_left(node)
                        parent = node.parent
                    parent.color = BLACK                  # case 3: zig-zig
                    grandparent.color = RED
                    self._rotate_right(grandparent)
            else:                                         # mirror image
                uncle = grandparent.left
                if _is_red(uncle):
                    parent.color = uncle.color = BLACK
                    grandparent.color = RED
                    node = grandparent
                else:
                    if node is parent.left:
                        node = parent
                        self._rotate_right(node)
                        parent = node.parent
                    parent.color = BLACK
                    grandparent.color = RED
                    self._rotate_left(grandparent)

        self.root.color = BLACK

    # ---------- deletion ----------

    def delete(self, key):
        """
        Input: key -- the value to remove.
        Output: True if the key was removed, False if it was not found.
        A node with two children takes its in-order successor's key, and the
        successor (which has at most one child) is removed instead. Removing
        a red node breaks nothing; removing a black node leaves one path
        short of a black node, which _delete_fixup repairs.
        """
        node = self.root
        while node is not None and node.key != key:
            node = node.left if key < node.key else node.right
        if node is None:
            return False  # key not in tree

        if node.left is not None and node.right is not None:
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key = successor.key
            node = successor

        child = node.left if node.left is not None else node.right
        parent = node.parent
        self._transplant(node, child)

        if node.color == BLACK:
            self._delete_fixup(child, parent)
        self.size -= 1
        return True

    def _delete_fixup(self, node, parent):
        """
        Input: node -- the node (possibly None) that replaced a removed black
                       node, and so carries an "extra black";
               parent -- node's parent (needed because node may be None).
        Output: None.
        Follows the four cases in CLRS Section 13.4, based on the colour of
        node's sibling and the sibling's children. Case 2 pushes the extra
        black up the tree; cases 3-4 end the loop with at most 2 rotations.
        """
        while node is not self.root and not _is_red(node):
            # When node is None, parent.left is None exactly when the removed
            # node was a left child (its sibling cannot be empty: it must
            # contain at least one black node to balance black-heights).
            if node is parent.left:
                sibling = parent.right
                if _is_red(sibling):                            # case 1
                    sibling.color = BLACK
                    parent.color = RED
                    self._rotate_left(parent)
                    sibling = parent.right
                if not _is_red(sibling.left) and not _is_red(sibling.right):
                    sibling.color = RED                         # case 2
                    node, parent = parent, parent.parent
                else:
                    if not _is_red(sibling.right):              # case 3
                        sibling.left.color = BLACK
                        sibling.color = RED
                        self._rotate_right(sibling)
                        sibling = parent.right
                    sibling.color = parent.color                # case 4
                    parent.color = BLACK
                    sibling.right.color = BLACK
                    self._rotate_left(parent)
                    node = self.root
            else:                                               # mirror image
                sibling = parent.left
                if _is_red(sibling):
                    sibling.color = BLACK
                    parent.color = RED
                    self._rotate_right(parent)
                    sibling = parent.left
                if not _is_red(sibling.left) and not _is_red(sibling.right):
                    sibling.color = RED
                    node, parent = parent, parent.parent
                else:
                    if not _is_red(sibling.left):
                        sibling.right.color = BLACK
                        sibling.color = RED
                        self._rotate_left(sibling)
                        sibling = parent.left
                    sibling.color = parent.color
                    parent.color = BLACK
                    sibling.left.color = BLACK
                    self._rotate_right(parent)
                    node = self.root

        if node is not None:
            node.color = BLACK