import abc
import math
import numpy as np
import logging
import pandas as pd
import os

from client.user_factory import UserFactory
from server.aggregator.aggregator_factory import AggregatorFactory
from server.query_handler.query_handler import QueryHelper
from server.hdtree import HDTree
from server.hdtree_ci import HDTreeCI


class ServerCI(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, users):
        logging.basicConfig(format='%(levelname)s:%(asctime)s: - server - :%(message)s',
                            level=logging.DEBUG)
        self.args = args
        self.query_handler = QueryHelper(args)
        self.users = users
        # self.aggregator = AggregatorFactory.create_aggregator(self.args.agg_method, self.args)
        self.aggregator = HDTreeCI(self.users, self.args)
        self.aggregator.gen_aux_info()
        self.aggregator.partition_users()

        self.raw_data = None
        self.sensitive_shard = None
        self.perturbed_shard = None
        self.domain_sizes = None

    def perturb_data(self):
        # logging.info('perturbing')
        self.raw_data = self.users.X
        column_index_to_perturb = [self.users.col_name_index_map[x] for x in self.users.column_to_perturb]
        self.sensitive_shard = self.raw_data[:, column_index_to_perturb]
        self.domain_sizes = [len(self.users.kv_map[x]) for x in self.users.column_to_perturb]
        self.perturbed_shard = self.aggregator.perturb(self.sensitive_shard)
        # logging.info('perturbed')

    # assume the func and agg_column is the same
    def process_query_all(self, queries):
        if self.perturbed_shard is None:
            self.perturb_data()

        node_indexs_list = []
        for query in queries:
            [func, agg_column, nodes] = self.query_handler.parse_query(query)
            node_indexs_list.append(self.translate_nodes(nodes))
        self.aggregator.load_queries(node_indexs_list)

        if func == 'sum':
            [result, true_result, total_weight, selectivity] = self.sum_handler_all(agg_column, node_indexs_list)
        elif func == 'avg':
            [result, true_result, total_weight, selectivity] = self.sum_handler_all(agg_column, node_indexs_list)
            [result, true_result, total_weight] = [result / self.args.n, true_result / self.args.n, total_weight]
        elif func == 'count':
            total_weight = 0
            [result, true_result] = self.count_handler(agg_column, node_indexs_list)

        return [result, true_result, total_weight, selectivity]

    def translate_nodes(self, nodes):
        if self.args.perturb_type in ['ord2cat2', 'ord1cat3q11', 'ord1cat3q12', 'ord1cat3q13', 'ord1cat3q02',
                                      'ord1lcat3q11', 'ord1lcat3q12', 'ord1lcat3q13', 'ord1lcat3q02',
                                      'ord2cat2q11', 'ord2cat2q12', 'ord4cat4q11', 'ord4cat4q12', 'ord4cat4q10',
                                      'ord4cat4q02', 'ord4cat4q01', 'ord4cat4q20', 'ord4cat4q21']:
            return self.translate_with_selectivity(nodes)
        else:
            return self.normal_translate(nodes)

    def translate_with_selectivity(self, nodes):
        node_indexs = []

        for node in nodes:
            (cols, vals) = node
            col_indexs = np.array([self.users.col_name_index_map[x] for x in cols])
            val_range_indexs = []
            for i in range(len(vals)):
                (l, r) = vals[i]
                if l == r:
                    l_index = self.users.vk_map[cols[i]][l]
                    r_index = self.users.vk_map[cols[i]][r] + 1
                else:
                    l_index, r_index = l, r
                    if cols[i] == 'INCTOT':
                        min = self.users.ranges[cols[i]]['min']
                        max = self.users.ranges[cols[i]]['max']
                        gran = self.users.ranges[cols[i]]['gran']

                        total_range = (max - min) / gran
                        interval = int(total_range * self.args.query_range)
                        l_index = np.random.choice(int(total_range - interval))
                        r_index = int(l_index + interval)

                val_range_indexs.append((l_index, r_index))
            ordering = np.argsort(col_indexs)
            col_indexs = np.array([col_indexs[x] for x in ordering])
            val_range_indexs = [val_range_indexs[x] for x in ordering]
            node_indexs.append((col_indexs, val_range_indexs))

        return node_indexs

    def normal_translate(self, nodes):
        node_indexs = []

        for node in nodes:
            (cols, vals) = node
            col_indexs = np.array([self.users.col_name_index_map[x] for x in cols])
            # val_indexs = np.array([self.users.vk_map[cols[i]][vals[i]] for i in range(len(col_indexs))])
            val_range_indexs = []
            for i in range(len(vals)):
                (l, r) = vals[i]
                if l == r:
                    l_index = self.users.vk_map[cols[i]][l]
                    r_index = self.users.vk_map[cols[i]][r] + 1
                    if not self.args.query_range == 0:
                        l_index = np.random.choice(len(self.users.vk_map[cols[i]]))
                        r_index = l_index + 1
                else:
                    l_index, r_index = l, r
                    # this can be used to translate. for now assume the queried values are fine grained
                    # l = pd.Interval(l, l + self.users.ranges[cols[i]]['gran'], closed='left')
                    # r = pd.Interval(r, r + self.users.ranges[cols[i]]['gran'], closed='left')
                    if not self.args.query_range == 0:
                        min = self.users.ranges[cols[i]]['min']
                        max = self.users.ranges[cols[i]]['max']
                        gran = self.users.ranges[cols[i]]['gran']

                        total_range = (max - min) / gran
                        interval = int(total_range * self.args.query_range)
                        l_index = np.random.choice(int(total_range - interval))
                        r_index = int(l_index + interval)

                val_range_indexs.append((l_index, r_index))

            # reorder
            ordering = np.argsort(col_indexs)
            col_indexs = np.array([col_indexs[x] for x in ordering])
            val_range_indexs = [val_range_indexs[x] for x in ordering]
            node_indexs.append((col_indexs, val_range_indexs))

        return node_indexs

    def count_handler(self, _, node_indexs):
        all_results, all_configs, all_multiples = self.aggregator.estimate_all(self.perturbed_shard, range(len(self.perturbed_shard)))
        est = self.aggregator.estimate_node(all_results, all_configs, all_multiples, node_indexs)
        tru = self.calc_nodes(self.sensitive_shard, node_indexs)
        return [est, tru]

    def sum_handler_all(self, agg_column, node_indexs_list):
        k_gran = self.users.ranges[agg_column]['gran']
        k_start = self.users.ranges[agg_column]['min']

        weights = []
        ests_list = [[] for _ in range(len(node_indexs_list))]
        trus_list = [[] for _ in range(len(node_indexs_list))]
        num_records = []
        for k, v in self.users.kv_map[agg_column].items():
            # logging.info('works on k=%s, v=%s' % (str(k), str(v)))

            weight = (k * k_gran + k_start)
            if weight == 0:
                if self.args.user_type == 'adult':
                    weight = 5
                elif self.args.user_type == 'bank':
                    weight = 3
                else:
                    weight = 50

            weights.append(weight)

            [perturbed_shard, sensitive_shard, fetched_index] = self.fetch_records(agg_column, k)
            num_records.append(len(perturbed_shard))
            logging.info('pid %d: weight: %s, fetched %s records' % (os.getpid(), str(weight), str(len(perturbed_shard))))
            if len(perturbed_shard) == 0:
                for i in range(len(node_indexs_list)):
                    ests_list[i].append(0)
                    trus_list[i].append(0)
                continue

            all_results, all_configs, all_multiples = self.aggregator.estimate_all(perturbed_shard, fetched_index)
            all_results, all_configs, all_multiples = self.aggregator.consist(len(perturbed_shard), all_results, all_configs, all_multiples)
            for i in range(len(node_indexs_list)):
                est = self.aggregator.estimate_node(all_results, all_configs, all_multiples, node_indexs_list[i])
                ests_list[i].append(est)
                tru = self.calc_nodes(sensitive_shard, node_indexs_list[i])
                trus_list[i].append(tru)
                logging.info('weight: %s: estimated value: %s; true value: %s' % (weight, str(est), str(tru)))

        new_weights = self.update_weights(weights, num_records)

        return [np.dot(ests_list, new_weights),
                np.dot(trus_list, weights),
                np.dot(num_records, weights),
                np.sum(trus_list, axis=1) / self.users.n]

    def update_weights(self, weights, num_records):

        if self.args.server_style == 'exact':
            new_weights = weights

        elif self.args.server_style == 'avg':
            new_weights = np.copy(weights)
            if self.args.num_bins < len(new_weights):
                temp_weights = np.array_split(new_weights, self.args.num_bins)
                temp_num = np.array_split(num_records, self.args.num_bins)
                count = 0
                for i in range(len(temp_weights)):
                    try:
                        new_weight = np.average(temp_weights[i], weights=temp_num[i])
                    except ZeroDivisionError:
                        new_weight = 0
                    # new_weight = 0 if bin_count == 0 else bin_total / bin_count
                    new_weights[count:count + len(temp_weights[i])] = new_weight
                    count += len(temp_weights[i])

        return new_weights

    def fetch_records(self, column, k):
        cond = self.raw_data[:, self.users.col_name_index_map[column]] == k
        perturbed_shard = []
        fetched_index = []
        for i in range(len(cond)):
            if cond[i]:
                fetched_index.append(i)
                perturbed_shard.append(self.perturbed_shard[i])
        return [perturbed_shard, self.sensitive_shard[cond], fetched_index]

    def estimate_nodes(self, perturbed_shard, node_indexs):
        result = 0
        hd_tree = self.aggregator.estimate_tree(perturbed_shard, self.domain_sizes)
        # logging.info('tree built')

        hd_tree_consist = self.aggregator.consist(hd_tree)
        # logging.info('tree consistent')

        for node_index in node_indexs:
            result += self.get_node_from_tree(hd_tree_consist, node_index)
        return result

    def calc_nodes(self, sensitive_shard, node_indexs):
        result = 0
        for node_index in node_indexs:
            (col_indexs, val_indexs) = node_index
            scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
            mask = np.logical_and(sensitive_shard[:, scol_indexs[0]] >= val_indexs[0][0], sensitive_shard[:, scol_indexs[0]] < val_indexs[0][1])
            for i in range(1, len(scol_indexs)):
                new_mask = np.logical_and(sensitive_shard[:, scol_indexs[i]] >= val_indexs[i][0], sensitive_shard[:, scol_indexs[i]] < val_indexs[i][1])
                mask = np.logical_and(mask, new_mask)
            result += np.sum(mask)
        return result

    def get_node_from_tree(self, tree, node_index):
        (col_indexs, val_indexs) = node_index
        d = len(col_indexs)
        scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
        return tree[d][tuple(scol_indexs)][tuple(val_indexs)]
