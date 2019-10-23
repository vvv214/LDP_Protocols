#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/3/18
    
"""
import math
import numpy as np
import logging
import itertools
from collections import OrderedDict
from server.fo.fo_factory import FOFactory


class HDTreeCI:

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

        self.gen_aux_info()

    def consist(self, n, all_results, all_configs, all_multiples):
        # todo: assume all layers have the same number of users, which is not true for rl
        # todo: assume 1 tree
        # todo: requires a complete tree for now
        tree_index = 0
        fanout = len(self.trees[tree_index].children)
        num_layer = self.tree_heights[tree_index]

        # leaf to root
        for i in range(len(all_configs) - 2, -1, -1):
            for est_i in range(len(all_results[i])):
                children_est = 0
                for child_i in range(fanout):
                    children_est += all_results[i + 1][(est_i * fanout + child_i, )]
                all_results[i][(est_i, )] = (fanout ** num_layer - fanout ** (num_layer - 1)) / (fanout ** num_layer - 1) * all_results[i][(est_i, )] + (fanout ** (num_layer - 1) - 1) / (fanout ** num_layer - 1) * children_est

        root_est = 0
        for est_i in range(len(all_results[0])):
            root_est += all_results[0][(est_i, )]
        print(n, root_est * all_multiples[0])

        # root to leaf
        for i in range(len(all_configs) - 1):
            for est_i in range(len(all_results[i])):
                children_est = 0
                for child_i in range(fanout):
                    children_est += all_results[i + 1][(est_i * fanout + child_i, )]

                diff = (all_results[i][(est_i, )] - children_est) / fanout
                for child_i in range(fanout):
                    all_results[i + 1][(est_i * fanout + child_i, )] += diff

        return all_results, all_configs, all_multiples

    @staticmethod
    def min_fanout(e):
        # min fanout does actually not depend on m
        m = 1024
        min_err = 1e6
        min_fanout = 1
        for i in range(2, m + 1, 1):
            layer = math.log(m, i)
            nom = math.exp(e / layer)
            denom = (math.exp(e / layer) - 1) ** 2
            err = (i - 1) * math.log(m, i) * nom / denom
            if err < min_err:
                min_err = err
                min_fanout = i
        return min_fanout

    def get_vol_sizes(self):
        vol_sizes = []
        for query_i in range(len(self.scol_indexs)):
            layer_indexs = self.layer_indexs[query_i]
            scol_indexs = self.scol_indexs[query_i]
            d = len(scol_indexs)
            vol_size = np.prod([len(layer_indexs[i]) for i in range(d)])
            vol_sizes.append(np.asscalar(vol_size))
        return vol_sizes

    def gen_aux_info(self):
        fanout = self.args.hie_fanout
        if fanout == 0 and self.args.layer_style == 'al':
            fanout = self.min_fanout(self.args.epsilon)
        for col in self.users.column_to_perturb:
            col_index = self.users.col_name_index_map[col]
            scol_index = self.users.colind_scolind_map[col_index]
            if col in self.users.ranges:
                c = int((self.users.ranges[col]['max'] - self.users.ranges[col]['min']) / self.users.ranges[col]['gran'])
                num_level = int(math.ceil(math.log(c, fanout))) + 1
                tree = self.build_range_tree(0, c + 1, fanout, 1, num_level)
                self.tree_heights[scol_index] = num_level
            else:
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

    # return a list of nodes whose location in the tree can be found via the range_node_map
    def range_to_tree_nodes(self, root, l, r):
        if self.args.layer_style == 'bl':
            nodes_to_invest = [root]
            bl = False
            while nodes_to_invest:
                new_nodes = []
                for node in nodes_to_invest:
                    if node.children:
                        new_nodes += node.children
                    else:
                        bl = True
                        break
                if bl:
                    break
                nodes_to_invest = new_nodes
        else:
            nodes_to_invest = [root]

        selected_nodes = []
        while nodes_to_invest:
            new_nodes = []
            for node in nodes_to_invest:
                # longer but (hopefully) clearer logic
                # if node.range[0] < l:
                #     if node.range[1] < l:
                #         continue
                #     elif l < node.range[1] < r:
                #         new_nodes += node.children
                #     else:
                #         new_nodes += node.children
                # elif l < node.range[0] < r:
                #     if node.range[1] < r:
                #         selected_nodes.append(node)
                #     else:
                #         new_nodes += node.children
                # else:
                #     continue

                if node.range[1] <= l or node.range[0] >= r:
                    continue

                if node.range[0] >= l and node.range[1] <= r:
                    selected_nodes.append(node)
                else:
                    new_nodes += node.children

            nodes_to_invest = new_nodes
        return selected_nodes

    @staticmethod
    def estimate_sr(perturbed_datas, fos, all_domain_sizes):
        n = len(perturbed_datas)
        dimension = len(all_domain_sizes)
        result = OrderedDict()

        for d in range(1, dimension + 1):

            result[d] = OrderedDict()
            attr_sets = itertools.combinations(range(dimension), d)

            for attr_set in attr_sets:
                result[d][attr_set] = OrderedDict()
                domain_size_subset = [range(all_domain_sizes[x]) for x in attr_set]
                all_possible_digits = itertools.product(*domain_size_subset)

                for digits in all_possible_digits:
                    same_count = 0
                    for i in range(n):
                        same = True
                        for j in range(len(attr_set)):
                            if not fos[attr_set[j]].support_sr(perturbed_datas[i][attr_set[j]], digits[j]):
                                same = False
                                break
                        same_count += same

                    minus_count = n * np.prod([fos[x].q for x in attr_set])
                    for d_ in range(1, d):
                        count_d_ = 0
                        chosen_sets_ = itertools.combinations(range(d), d_)
                        for chosen_set_ in chosen_sets_:
                            coeff = np.prod([fos[d].q for d in attr_set])
                            coeff *= np.prod(
                                [(fos[attr_set[x]].p / fos[attr_set[x]].q - 1) for x in chosen_set_])

                            attr_set_ = tuple([attr_set[chosen_set_[k]] for k in range(d_)])
                            digits_ = tuple([digits[chosen_set_[k]] for k in range(d_)])

                            count_d_ += result[d_][attr_set_][digits_] * coeff
                        minus_count += count_d_

                    est = (same_count - minus_count) / np.prod([(fos[x].p - fos[x].q) for x in attr_set])
                    result[d][attr_set][digits] = est

        return result

    @staticmethod
    def aggregate_ce(precise_result, all_domain_sizes):
        domain_sizes = []
        for i in range(len(all_domain_sizes)):
            for j in range(len(all_domain_sizes[i])):
                domain_sizes.append(all_domain_sizes[i][j])
        dimension = len(domain_sizes)
        result = OrderedDict()
        result[dimension] = OrderedDict()
        result[dimension][tuple(range(dimension))] = precise_result

        for d in range(dimension - 1, 0, -1):
            result[d] = OrderedDict()
            attr_sets = itertools.combinations(range(dimension), d)

            for attr_set in attr_sets:
                result[d][attr_set] = OrderedDict()
                domain_size_subset = [range(domain_sizes[x]) for x in attr_set]
                all_possible_digits = itertools.product(*domain_size_subset)

                for dim in range(d):
                    if not attr_set[dim] == dim:
                        break
                if attr_set[-1] == d - 1:
                    dim = d
                remain_attr = insert_index = dim

                for digits in all_possible_digits:
                    count = 0
                    for v in range(domain_sizes[dim]):
                        new_attr_set = list(attr_set)
                        new_attr_set.insert(insert_index, remain_attr)
                        new_digits = list(digits)
                        new_digits.insert(insert_index, v)

                        count += result[d + 1][tuple(new_attr_set)][tuple(new_digits)]
                    result[d][attr_set][digits] = count
        return result

    def estimate_tree(self, perturbed_datas, attr_config, layer_config, group_fos):
        all_domain_sizes = []
        sr_domain_sizes = []
        attr_count = 0
        for report_attrs in attr_config:
            domain_sizes = []
            for report_attr in report_attrs:
                domain_sizes.append(len(self.tree_range_lists[report_attr][layer_config[attr_count]]))
                attr_count += 1
            all_domain_sizes.append(domain_sizes)
            sr_domain_sizes.append(np.prod(domain_sizes))

        result_after_ce = self.estimate_sr(perturbed_datas, group_fos, sr_domain_sizes)
        precise_result = self.expand_ce(result_after_ce, all_domain_sizes, sr_domain_sizes)
        return precise_result

    @staticmethod
    def expand_ce(result, all_domain_sizes, sr_domain_sizes):
        result = result[len(sr_domain_sizes)][tuple(range(len(sr_domain_sizes)))]
        new_result = OrderedDict()

        all_domains = []
        for i in range(len(all_domain_sizes)):
            for j in range(len(all_domain_sizes[i])):
                all_domains.append(range(all_domain_sizes[i][j]))
        all_possible_digits = itertools.product(*all_domains)
        sr_domains = [range(sr_domain_sizes[i]) for i in range(len(all_domain_sizes))]
        sr_possible_digits = itertools.product(*sr_domains)
        for digits in all_possible_digits:
            new_result[digits] = result[next(sr_possible_digits)]
        return new_result

    # assume the partition is ordered
    def estimate_all(self, perturbed_datas, selected_indexs):
        total_n = len(perturbed_datas)
        selected_i = 0
        all_results = []
        all_configs = []
        all_multiples = []
        for group_index in range(len(self.group_indexs)):
            (user_start, user_end) = self.group_indexs[group_index]
            old_selected_i = selected_i
            while selected_i < len(selected_indexs) and selected_indexs[selected_i] < user_end:
                selected_i += 1

            if selected_i > old_selected_i:
                selected_perturbed_datas = perturbed_datas[old_selected_i:selected_i]
                config_n = len(selected_perturbed_datas)
                (attr_config, layer_config) = self.group_configs[group_index]
                group_fos = self.group_fos[group_index]
                if self.args.layer_style == 'al':
                    all_attrs = []
                    for i in range(len(attr_config)):
                        for j in range(len(attr_config[i])):
                            all_attrs.append(attr_config[i][j])

                    all_heights = [range(self.tree_heights[x]) for x in all_attrs]
                    all_all_layers = itertools.product(*all_heights)

                    cluster_heights = [range(np.prod([self.tree_heights[x] for x in attr_config[i]])) for i in range(len(attr_config))]
                    report_counts = itertools.product(*cluster_heights)
                    for layer_config in all_all_layers:
                        report_count = next(report_counts)
                        exist_star = False
                        for count in report_count:
                            if count == 0:
                                exist_star = True
                                break
                        # some report does not contribute info, which is prohibited
                        if exist_star:
                            continue

                        data_to_estimate = [[0 for _ in range(len(attr_config))] for _ in
                                            range(len(selected_perturbed_datas))]
                        for i in range(len(selected_perturbed_datas)):
                            for j in range(len(attr_config)):
                                data_to_estimate[i][j] = selected_perturbed_datas[i][j][report_count[j] - 1]
                        fos = []
                        for i in range(len(attr_config)):
                            fos.append(group_fos[i][report_count[i] - 1])
                        results = self.estimate_tree(data_to_estimate, attr_config, layer_config, fos)
                        all_results.append(results)
                        all_configs.append((attr_config, layer_config))
                        all_multiples.append(total_n / config_n)
                else:
                    fos = [group_fos[i][0] for i in range(len(group_fos))]
                    data_to_estimate = [[0 for _ in range(len(attr_config))] for _ in range(len(selected_perturbed_datas))]
                    for i in range(len(selected_perturbed_datas)):
                        for j in range(len(attr_config)):
                            data_to_estimate[i][j] = selected_perturbed_datas[i][j][0]
                    results = self.estimate_tree(data_to_estimate, attr_config, layer_config, fos)
                    all_results.append(results)
                    all_configs.append((attr_config, layer_config))
                    all_multiples.append(total_n / config_n)
        return all_results, all_configs, all_multiples

    def load_queries(self, bl_node_index_list):
        self.num_tree_nodes = []
        self.layer_indexs = []
        self.scol_indexs = []

        for query_i in range(len(bl_node_index_list)):
            bl_node_indexs = bl_node_index_list[query_i]
            # todo: for now, assume it is continuous and thus contains only one item
            bl_node_index = bl_node_indexs[0]
            (col_indexs, val_range_indexs) = bl_node_index
            d = len(col_indexs)
            total_d = len(self.users.column_to_perturb)
            scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
            remaining_indexs = list(set(range(total_d)).difference(set(scol_indexs)))
            scol_indexs += remaining_indexs
            layer_indexs = [[] for _ in scol_indexs]
            for i in range(total_d):
                if i < d:
                    attr = scol_indexs[i]
                    (l, r) = val_range_indexs[i]
                    if r == l + 1:
                        layer_index_tuple = (self.tree_heights[attr] - 1, l)
                        layer_indexs[i].append(layer_index_tuple)
                    else:
                        tree_nodes_to_est = self.range_to_tree_nodes(self.trees[attr], l, r)
                        for node in tree_nodes_to_est:
                            loc_list = self.tree_range_node_map[attr][node.range]
                            # for now assume the tree is perfect so there is only one location for each range
                            layer_index_tuple = loc_list[-1]
                            layer_indexs[i].append(layer_index_tuple)
                else:
                    # this attribute is not queried
                    layer_indexs[i].append((0, 0))

            # reorder
            ordering = np.argsort(scol_indexs)
            scol_indexs = np.array([scol_indexs[x] for x in ordering])
            layer_indexs = [layer_indexs[x] for x in ordering]

            self.num_tree_nodes.append([range(len(layer_indexs[i])) for i in range(total_d)])
            self.layer_indexs.append(layer_indexs)
            self.scol_indexs.append(scol_indexs)

    def estimate_node(self, all_results, all_configs, all_multiples, bl_node_indexs):
        bl_node_index = bl_node_indexs[0]
        (col_indexs, val_range_indexs) = bl_node_index
        d = len(col_indexs)
        scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
        layer_indexs = [[] for _ in scol_indexs]
        for i in range(d):
            attr = scol_indexs[i]
            (l, r) = val_range_indexs[i]
            if r == l:
                layer_index_tuple = (self.tree_heights[attr] - 1, l)
                layer_indexs[i].append(layer_index_tuple)
            else:
                tree_nodes_to_est = self.range_to_tree_nodes(self.trees[attr], l, r)
                for node in tree_nodes_to_est:
                    loc_list = self.tree_range_node_map[attr][node.range]
                    # for now assume the tree is perfect so there is only one location for each range
                    layer_index_tuple = loc_list[-1]
                    layer_indexs[i].append(layer_index_tuple)

        num_tree_nodes = [range(len(layer_indexs[i])) for i in range(d)]
        all_possible_tree_nodes = itertools.product(*num_tree_nodes)
        query_result = 0
        for tree_node in all_possible_tree_nodes:
            node_ests = []
            for config_index in range(len(all_configs)):
                (attr_config, layer_config) = all_configs[config_index]
                val_indexs = []
                match = True
                for i in range(len(scol_indexs)):
                    attr = scol_indexs[i]
                    (layer, val_index) = layer_indexs[i][tree_node[i]]
                    val_indexs.append(val_index)
                    if not layer_config[attr] == layer:
                        match = False
                        break
                if match:
                    node_ests.append(all_results[config_index][tuple(val_indexs)] * all_multiples[config_index])
            # todo: use average for now, should use weighted average
            if len(node_ests) == 0:
                node_est = 0
            else:
                node_est = np.mean(node_ests)
            query_result += node_est
        return query_result

    def perturb(self, datas):
        perturbed_datas = [[] for _ in range(len(datas))]
        for group_index in range(len(self.group_indexs)):
            (user_start, user_end) = self.group_indexs[group_index]
            (attr_config, layer_config) = self.group_configs[group_index]
            num_report = len(attr_config)
            eps = self.args.epsilon / num_report

            group_perturbed_datas = [[[] for _ in range(num_report)] for _ in range(user_end - user_start)]
            config_fos = []
            attr_count = 0
            for report_count in range(len(attr_config)):
                report_attrs = attr_config[report_count]
                report_layers = layer_config[attr_count: attr_count + len(report_attrs)]
                attr_count += len(report_attrs)
                report_fos = []
                if self.args.layer_style == 'al':
                    report_heights = [range(self.tree_heights[x]) for x in report_attrs]
                    num_report_layers = np.prod([self.tree_heights[x] for x in report_attrs]) - 1
                    all_report_layers = itertools.product(*report_heights)
                    next(all_report_layers)
                    for report_layers in all_report_layers:
                        # domain_size = np.prod([len(self.tree_range_lists[report_attrs[j]][report_layers[j]]) for j in
                        #                        range(len(report_attrs))])
                        # fo = FOFactory.create_fo('adap', eps / num_level_comb, domain_size)
                        fo = self.atom_perturb(eps / num_report_layers, group_perturbed_datas, report_count, datas, user_start, user_end, report_attrs, report_layers)
                        report_fos.append(fo)
                else:
                    fo = self.atom_perturb(eps, group_perturbed_datas, report_count, datas, user_start,
                                           user_end, report_attrs, report_layers)
                    report_fos.append(fo)
                config_fos.append(report_fos)

            for i in range(user_end - user_start):
                perturbed_datas[user_start + i] = group_perturbed_datas[i]
            self.group_fos.append(config_fos)

            # for report_index in range(len(num_report)):
            #     report_attrs = attr_config[report_index]
            #     num_attr = len(report_attrs)
            #     attrs = []
            #     for attr_index in range(len(num_attr)):
            #         (attr, layer) = report_attrs[attr_index]
            #         attrs.append(attr)
            #         if attr in self.users.num_column and layer < self.tree_heights[attr]:
            #             bins = [self.tree_range_lists[layer][0][0]] + [x[1] for x in self.tree_range_lists[layer]]
            #             group_datas[:, attr] = np.digitize(group_datas[:, attr], bins) - 1
            #     group_perturbed_datas[:, report_index] = fos[report_index].perturb(group_datas[:, attrs])
            # perturbed_datas += group_perturbed_datas
        return perturbed_datas

    def atom_perturb(self, eps, group_perturbed_datas, report_count, datas, user_start, user_end, report_attrs, report_layers):
        domain_sizes = [0] * len(report_attrs)
        group_datas = datas[user_start:user_end, report_attrs].copy()
        for i in range(len(report_attrs)):
            report_attr = report_attrs[i]
            report_layer = report_layers[i]
            bins = [self.tree_range_lists[report_attr][report_layer][0][0]] + [x[1] for x in self.tree_range_lists[report_attr][report_layer]]
            # domain_sizes[i] = len(bins) - 1
            domain_sizes[i] = len(self.tree_range_lists[report_attrs[i]][report_layers[i]])
            group_datas[:, i] = np.digitize(group_datas[:, i], bins) - 1

        total_size = np.prod(domain_sizes)
        fo = FOFactory.create_fo('adap', self.args, eps, total_size)

        datas_to_perturb = np.zeros(len(group_datas), dtype=np.int)
        for domain in range(len(domain_sizes)):
            factor = int(np.prod(domain_sizes[domain + 1:]))
            datas_to_perturb += group_datas[:, domain] * factor
        perturbed_datas = fo.perturb(datas_to_perturb, np.prod(domain_sizes))

        for i in range(len(perturbed_datas)):
            group_perturbed_datas[i][report_count].append(perturbed_datas[i])
        return fo

    # fills data structures: group_indexs, group_configs, and group_fos
    # each config is a tuple of attr_config and layer_config;
    #   attr_config is grouped into reports, while layer_config is not
    #   e.g., (((1,1),(2,1)),(3,1,4,1))
    # each group_fos is a set of report_fos which is a list of fos for each layer combination
    # the users are partitioned into chunks (should shuffle first), and
    #   group_indexs is the list of tuples of start and end indexs for this group
    def partition_users(self):
        num_user = self.users.n
        dimension = len(self.users.column_to_perturb)
        # todo: for now, assume all attributes are used
        d = dimension
        # this solution is brute force
        # more elegant solutions needed!
        configs = []
        attr_configs = {}
        report_size = min(d, self.args.ce_size)
        report_per_user = int(math.ceil(d / report_size))
        used_dim_set = itertools.combinations(range(dimension), d)
        for used_dims in used_dim_set:

            perms = itertools.permutations(used_dims)
            for perm in perms:
                # attributes are sorted
                attr_config = []
                for i in range(report_per_user):
                    sr_set = sorted(perm[i * report_size: (i + 1) * report_size])
                    attr_config.append(tuple(sr_set))
                attr_config = tuple(sorted(attr_config))

                if attr_config in attr_configs:
                    continue
                attr_configs[attr_config] = 1
                num_layers = [range(self.tree_heights[x]) for x in perm]

                # fuse the layers into configuration
                if self.args.layer_style == 'rl':
                    layer_configs = itertools.product(*num_layers)
                    cluster_heights = [range(np.prod([self.tree_heights[x] for x in attr_config[i]])) for i in
                                       range(len(attr_config))]
                    report_counts = itertools.product(*cluster_heights)
                    for layer_config in layer_configs:
                        report_count = next(report_counts)
                        exist_star = False
                        for count in report_count:
                            if count == 0:
                                exist_star = True
                                break
                        # some report does not contribute info, which is prohibited
                        if exist_star:
                            continue
                        configs.append(tuple([attr_config, layer_config]))
                elif self.args.layer_style == 'bl':
                    configs.append(tuple([attr_config, tuple([self.tree_heights[x] - 1 for x in perm])]))
                elif self.args.layer_style == 'al':
                    configs.append(tuple([attr_config, tuple([self.tree_heights[x] - 1 for x in perm])]))
                    # layer_configs = itertools.product(*num_layers)
                    # next(layer_configs)
                    # mat_layer_configs = []
                    # for layer_config in layer_configs:
                    #     mat_layer_configs.append(layer_config)
                    # configs.append(tuple([attr_config, mat_layer_configs]))

            num_group = len(configs)
            self.group_configs = configs
            for group_id in range(num_group):
                user_start = int(num_user / num_group * group_id)
                user_end = int(num_user / num_group * (group_id + 1))
                self.group_indexs.append((user_start, user_end))


class Tree(object):
    def __init__(self):
        self.children = []
        self.range = None
