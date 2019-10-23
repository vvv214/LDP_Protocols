class Tree(object):
    def __init__(self):
        self.children = []
        self.range = None


# the tree is actually a list (for layers) of list of intervals (denoted using tuples)
# left is closed, right is open, i.e., [l, r)
def build_range_tree(self, l, r, fanout, cur_layer, total_layer):
    tree = Tree()
    tree.range = (l, r)
    if cur_layer == total_layer:
        return tree

    step_size = (r - l) / fanout

    cur_l = l
    for i in range(fanout):
        cur_r = l + int(step_size * (i + 1))
        if not cur_r == cur_l:
            tree.children.append(self.build_range_tree(cur_l, cur_r, fanout, cur_layer + 1, total_layer))
            cur_l = cur_r

    return tree


# range_lists: dict of ranges indexed by layer
#   e.g., {0: [(0,8)], 1: [(0,4),(5,8)], etc}
# range_node_map: map from a node range to the location of the node in the tree (range_lists)
#   e.g., {(0,8): [(0,0)], (0,4): [(1,0)], etc}
# todo:
#   if the tree is not perfect, each node can appear multiple times.
#   for now, assume the values are partitioned so that the tree is perfect
@staticmethod
def traverse_range_tree(root):
    range_lists = {0: [root.range]}
    range_node_map = {root.range: [(0, 0)]}
    layer_nodes = root.children
    layer = 1
    while layer_nodes:
        index = 0
        range_lists[layer] = []
        new_nodes = []
        for node in layer_nodes:
            range_lists[layer].append(node.range)
            if node.range in range_node_map:
                range_node_map[node.range].append((layer, index))
            else:
                range_node_map[node.range] = [(layer, index)]
            index += 1
            new_nodes += node.children
        layer_nodes = new_nodes
        layer += 1
    return range_lists, range_node_map
