#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/3/18
    
"""
import logging
import math
import itertools
import numpy as np


class HDTreeMG:

    def __init__(self, users, args):
        logging.basicConfig(format='%(levelname)s:%(asctime)s: - %(name)s - :%(message)s',
                            level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.args = args
        self.users = users

        # related to hd_tree
        self.trees = {}
        self.tree_heights = {}
        self.tree_range_lists = {}
        self.tree_range_node_map = {}

        # related to user partition
        self.group_indexs = []
        self.group_configs = []
        self.group_fos = []

        # related to queries
        self.num_tree_nodes = []
        self.layer_indexs = []
        self.scol_indexs = []

        # new to marginal
        self.sensitive_data = []
        self.estimates = {}
        self.view_config = {
            # for adult, debug purpose
            # 'ord2cat2': [[0], [2], [1, 3]],
            # ipums
            'ord1': [[0]],
            'ord2': [[0, 1]],
            # 'ord3': [[0], [1], [2]],
            # 'ord3': [[0, 1, 2]],
            # 'ord3': [[0, 1, 2]],
            'ord3': [[0, 1], [0, 2], [1, 2], ],
            'ord2cat2': [[0, 2], [1], [3]],
            'ord4cat4': [[0, 2, 3, 4], [1], [5], [6], [7]],
        }
        self.queries = None

        self.gen_aux_info()

    def gen_aux_info(self):
        for col in self.users.column_to_perturb:
            col_index = self.users.col_name_index_map[col]
            scol_index = self.users.colind_scolind_map[col_index]
            tree = self.build_flat_tree(col)
            self.tree_heights[scol_index] = 2
            self.trees[scol_index] = tree
            self.tree_range_lists[scol_index], self.tree_range_node_map[scol_index] = self.traverse_range_tree(tree)

    def build_flat_tree(self, col):
        tree = Tree()
        domain_size = len(self.users.kv_map[col])
        tree.range = (0, domain_size)
        for i in range(domain_size):
            leaf = Tree()
            leaf.range = (i, i + 1)
            tree.children.append(leaf)
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

    def load_queries(self, bl_node_index_list):
        self.queries = []
        self.num_tree_nodes = []
        self.layer_indexs = []
        self.scol_indexs = []

        for query_i in range(len(bl_node_index_list)):
            bl_node_indexs = bl_node_index_list[query_i]
            # todo: for now, assume it is continuous and thus contains only one item
            bl_node_index = bl_node_indexs[0]
            (col_indexs, val_range_indexs) = bl_node_index
            scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
            self.queries.append((scol_indexs, val_range_indexs))

    def get_vol_sizes(self):
        vol_sizes = []
        for query_i in range(len(self.scol_indexs)):
            layer_indexs = self.layer_indexs[query_i]
            scol_indexs = self.scol_indexs[query_i]
            d = len(scol_indexs)
            vol_size = np.prod([len(layer_indexs[i]) for i in range(d)])
            vol_sizes.append(np.asscalar(vol_size))
        return vol_sizes

    def combined_encoding(self, cols, values):
        layer_domains = []
        for col in cols:
            layer_domains.append(len(self.tree_range_lists[col][self.tree_heights[col] - 1]))
        ce_values = np.zeros(len(values))
        for i, col in enumerate(cols):
            factor = int(np.prod(layer_domains[i + 1:]))
            ce_values += values[:, col] * factor
        return ce_values, np.prod(layer_domains)

    @staticmethod
    def tally(ce_report, domain_size):
        return np.histogram(ce_report, bins=range(domain_size + 1))[0]

    @staticmethod
    def post_process(estimates, n):
        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            estimates[estimates < 0] = 0
            total = sum(estimates)
            mask = estimates > 0
            diff = (n - total) / sum(mask)
            estimates[mask] += diff
        return estimates

    @staticmethod
    def optimized_unary_encoding(epsilon, original_vector, num_users):
        p = 0.5
        q = 1 / (1 + math.exp(epsilon))

        # step 1: initial count
        X = np.copy(original_vector)
        ESTIMATE_X = np.random.binomial(X, p)

        # step 2: noise
        X = np.copy(original_vector)
        X = num_users - X
        ESTIMATE_X += np.random.binomial(X, q)

        # step 3: normalize
        a = 1.0 / (p - q)
        b = num_users * q / (p - q)

        return a * ESTIMATE_X - b

    def estimate_query(self, perturbed_datas, selected_indexs, query_i):
        if len(self.args.perturb_type) < 8:
            views = self.view_config[self.args.perturb_type[:4]]
        else:
            views = self.view_config[self.args.perturb_type[:8]]

        if tuple(selected_indexs) not in self.estimates:
            num_group = len(views)
            group_indexes = np.copy(selected_indexs)
            np.random.shuffle(group_indexes)
            group_size_frac = len(selected_indexs) / num_group
            groups_est = [0] * len(views)
            for group_i, attrs in enumerate(views):
                group_report = self.sensitive_data[
                    group_indexes[int(group_size_frac * group_i): int(group_size_frac * (group_i + 1))]]
                ce_report, domain_size = self.combined_encoding(views[group_i], group_report)
                vector = self.tally(ce_report, domain_size)
                marginal_est = self.optimized_unary_encoding(self.args.epsilon, vector, len(ce_report))
                # marginal_est = self.post_process(marginal_est, len(group_report))
                marginal_est /= len(group_report)
                groups_est[group_i] = marginal_est
            self.estimates[tuple(selected_indexs)] = groups_est

        # some of the following can be put to the function load_query
        (scol_indexs, val_range_indexs) = self.queries[query_i]
        ests = np.zeros(len(views))
        for group_i, attrs in enumerate(views):
            estimates = self.estimates[tuple(selected_indexs)][group_i]
            if len(attrs) == 1:
                selected_i = -1
                for i, scol in enumerate(scol_indexs):
                    if scol == attrs[0]:
                        selected_i = i
                        break
                if selected_i == -1:
                    ests[group_i] = 1
                else:
                    (l, r) = val_range_indexs[selected_i]
                    ests[group_i] = sum(estimates[l: r])
                    if ests[group_i] == 0:
                        break
            else:
                query = []
                for attr in attrs:
                    selected_i = -1
                    for i, scol in enumerate(scol_indexs):
                        if attr == scol:
                            selected_i = i
                            break
                    if selected_i == -1:
                        query.append(range(len(self.tree_range_lists[attr][self.tree_heights[attr] - 1])))
                    else:
                        (l, r) = val_range_indexs[selected_i]
                        if r - l > 1:
                            query.append(range(l, r))
                        else:
                            query.append([l])
                combined_queries = []
                full_query = np.zeros((1, len(self.tree_heights)))
                for q in itertools.product(*query):
                    full_query[0, attrs] = q
                    combined_queries.append(self.combined_encoding(attrs, full_query)[0])
                combined_queries = np.array(combined_queries, dtype=np.int)
                ests[group_i] = sum(estimates[combined_queries])
                if ests[group_i] == 0:
                    break
        if self.args.perturb_type == 'ord3q2':
            return ests[2] * len(perturbed_datas)
        # return np.prod(ests) * len(perturbed_datas)
        return self.synthesize_results(ests, query_i, views, len(perturbed_datas))

    def synthesize_results(self, ests, query_i, views, data_size):
        (scol_indexs, val_range_indexs) = self.queries[query_i]
        for group_i, attrs in enumerate(views):
            for i, scol in enumerate(scol_indexs):
                if scol not in attrs:
                    selected_i = i
                    break
            (l, r) = val_range_indexs[selected_i]
            ests[group_i] *= (r - l) / len(self.tree_range_lists[scol][self.tree_heights[scol] - 1])

        query_result = np.mean(ests) * data_size
        return query_result

    def perturb(self, datas):
        self.sensitive_data = np.array(datas)
        return datas

    def partition_users(self):
        return


class Tree(object):
    def __init__(self):
        self.children = []
        self.range = None
