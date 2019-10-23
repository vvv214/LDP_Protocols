import abc
import logging

import numpy as np

from server.hdtree import HDTree
from server.hdtree_mg import HDTreeMG
from server.query_handler.query_handler import QueryHelper


class Server(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, users):
        logging.basicConfig(format='%(levelname)s:%(asctime)s: - server - :%(message)s',
                            level=logging.DEBUG)
        self.args = args
        self.query_handler = QueryHelper(args)
        self.users = users
        # self.aggregator = AggregatorFactory.create_aggregator(self.args.agg_method, self.args)
        self.aggregator = HDTree(self.users, self.args)
        if self.args.layer_style == 'mg':
            self.aggregator = HDTreeMG(self.users, self.args)
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

        # func = 'count'
        if func == 'sum':
            [result, true_result, total_weight, selectivity] = self.sum_handler_all(agg_column, node_indexs_list)
        elif func == 'avg':
            [result, true_result, total_weight, selectivity] = self.avg_handler_all(agg_column, node_indexs_list)
        elif func == 'count':
            [result, true_result, total_weight, selectivity] = self.count_handler_all(agg_column, node_indexs_list)

        else:
            raise ValueError('func wrong: %s' % func)

        return [result, true_result, total_weight, selectivity]

    def process_query_all_new(self, queries):
        if self.perturbed_shard is None:
            self.perturb_data()

        node_indexs_list = []
        for query in queries:
            [func, agg_column, nodes] = self.query_handler.parse_query(query)
            node_indexs_list.append(self.translate_nodes(nodes))
        self.aggregator.load_queries(node_indexs_list)

        return self.all_handler_all(agg_column, node_indexs_list)

    def process_query(self, query):
        if self.perturbed_shard is None:
            self.perturb_data()

        [func, agg_column, nodes] = self.query_handler.parse_query(query)
        node_indexs = self.translate_nodes(nodes)
        if func == 'sum':
            [result, true_result] = self.sum_handler(agg_column, node_indexs)
        elif func == 'avg':
            [result, true_result] = self.avg_handler(agg_column, node_indexs)
        elif func == 'count':
            [result, true_result] = self.count_handler(agg_column, node_indexs)

        return [result, true_result]

    def translate_nodes(self, nodes):
        if self.args.perturb_type in ['ord2cat2', 'ord2cat2q11', 'ord2cat2q20', 'ord2cat2q22', 'ord2cat2q10',
                                      'ord4cat4q11', 'ord4cat4q12', 'ord4cat4q10', 'ord4cat4q02', 'ord4cat4q01',
                                      'ord4cat4q20', 'ord4cat4q21']:
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
                    if cols[i] == 'INCTOT' or (cols[i] == 'FTOTINC' and self.args.vary == 'd'):
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
            num_ord_dim = int(self.args.perturb_type[3])
            query_range = self.args.query_range ** (1.0 / num_ord_dim)
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
                        interval = int(total_range * query_range)
                        l_index = np.random.choice(int(total_range - interval))
                        r_index = int(l_index + interval)

                val_range_indexs.append((l_index, r_index))

            # reorder
            ordering = np.argsort(col_indexs)
            col_indexs = np.array([col_indexs[x] for x in ordering])
            val_range_indexs = [val_range_indexs[x] for x in ordering]
            node_indexs.append((col_indexs, val_range_indexs))

        return node_indexs

    def all_handler_all(self, agg_column, node_indexs_list):
        k_gran = self.users.ranges[agg_column]['gran']
        k_start = self.users.ranges[agg_column]['min']

        weights = []
        ests_list = [[] for _ in range(len(node_indexs_list))]
        trus_list = [[] for _ in range(len(node_indexs_list))]
        num_records = []
        for k, v in self.users.kv_map[agg_column].items():
            # logging.info('works on k=%s, v=%s' % (str(k), str(v)))

            weight = (k * k_gran + k_start)
            if not self.args.vary == 'none':
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
            # logging.info('%d: weight: %s, fetched %s records' % (os.getpid(), str(weight), str(len(perturbed_shard))))
            if len(perturbed_shard) == 0:
                for i in range(len(node_indexs_list)):
                    ests_list[i].append(0)
                    trus_list[i].append(0)
                continue

            for i in range(len(node_indexs_list)):
                est = self.aggregator.estimate_query(perturbed_shard, fetched_index, i)
                ests_list[i].append(est)
                tru = self.calc_nodes(sensitive_shard, node_indexs_list[i])
                trus_list[i].append(tru)
                if self.args.exp_round == 1:
                    logging.info('weight: %s: estimated value: %s; true value: %s' % (weight, str(est), str(tru)))

        new_weights = self.update_weights(weights, num_records)

        # sum, sum_true, total_weight, selectivity, avg, avg_true, count, count_true
        return [np.dot(ests_list, new_weights),
                np.dot(trus_list, weights),
                np.dot(num_records, weights),
                np.sum(trus_list, axis=1) / self.users.n,
                np.dot(ests_list, new_weights) / np.sum(ests_list, axis=1),
                np.dot(trus_list, weights) / np.sum(trus_list, axis=1),
                np.sum(ests_list, axis=1),
                np.sum(trus_list, axis=1)
                ]

    def count_handler(self, _, node_indexs):
        all_results, all_configs, all_multiples = self.aggregator.estimate_all(self.perturbed_shard,
                                                                               range(len(self.perturbed_shard)))
        est = self.aggregator.estimate_node(all_results, all_configs, all_multiples, node_indexs)
        tru = self.calc_nodes(self.sensitive_shard, node_indexs)
        return [est, tru]

    def count_handler_all(self, agg_column, node_indexs_list):

        weights = []
        ests_list = [[] for _ in range(len(node_indexs_list))]
        trus_list = [[] for _ in range(len(node_indexs_list))]
        num_records = []

        weight = 1

        weights.append(weight)

        [perturbed_shard, sensitive_shard, fetched_index] = [self.perturbed_shard, self.sensitive_shard,
                                                             range(len(self.sensitive_shard))]
        num_records.append(len(perturbed_shard))

        for i in range(len(node_indexs_list)):
            est = self.aggregator.estimate_query(perturbed_shard, fetched_index, i)
            ests_list[i].append(est)
            tru = self.calc_nodes(sensitive_shard, node_indexs_list[i])
            trus_list[i].append(tru)

        new_weights = self.update_weights(weights, num_records)

        return [np.dot(ests_list, new_weights),
                np.dot(trus_list, weights),
                np.dot(num_records, weights),
                np.sum(trus_list, axis=1) / self.users.n]

    def avg_handler_all(self, agg_column, node_indexs_list):
        k_gran = self.users.ranges[agg_column]['gran']
        k_start = self.users.ranges[agg_column]['min']

        weights = []
        ests_list = [[] for _ in range(len(node_indexs_list))]
        trus_list = [[] for _ in range(len(node_indexs_list))]
        num_records = []
        for k, v in self.users.kv_map[agg_column].items():
            # logging.info('works on k=%s, v=%s' % (str(k), str(v)))

            weight = (k * k_gran + k_start)
            if not self.args.vary == 'none':
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
            # logging.info('%d: weight: %s, fetched %s records' % (os.getpid(), str(weight), str(len(perturbed_shard))))
            if len(perturbed_shard) == 0:
                for i in range(len(node_indexs_list)):
                    ests_list[i].append(0)
                    trus_list[i].append(0)
                continue

            for i in range(len(node_indexs_list)):
                est = self.aggregator.estimate_query(perturbed_shard, fetched_index, i)
                ests_list[i].append(est)
                tru = self.calc_nodes(sensitive_shard, node_indexs_list[i])
                trus_list[i].append(tru)
                # logging.info('weight: %s: estimated value: %s; true value: %s' % (weight, str(est), str(tru)))

        new_weights = self.update_weights(weights, num_records)

        if self.args.vary == 'none':
            print(np.dot(ests_list, new_weights))
            print(np.sum(ests_list, axis=1))
            print(np.dot(ests_list, new_weights) / np.sum(ests_list, axis=1))

            print()

            print(np.dot(trus_list, new_weights))
            print(np.sum(trus_list, axis=1))
            print(np.dot(trus_list, new_weights) / np.sum(trus_list, axis=1))

        return [np.dot(ests_list, new_weights) / np.sum(ests_list, axis=1),
                np.dot(trus_list, weights) / np.sum(trus_list, axis=1),
                np.dot(num_records, weights),
                np.sum(trus_list, axis=1) / self.users.n]

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
            if not self.args.vary == 'none':
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
            # logging.info('%d: weight: %s, fetched %s records' % (os.getpid(), str(weight), str(len(perturbed_shard))))
            if len(perturbed_shard) == 0:
                for i in range(len(node_indexs_list)):
                    ests_list[i].append(0)
                    trus_list[i].append(0)
                continue

            for i in range(len(node_indexs_list)):
                est = self.aggregator.estimate_query(perturbed_shard, fetched_index, i)
                ests_list[i].append(est)
                tru = self.calc_nodes(sensitive_shard, node_indexs_list[i])
                trus_list[i].append(tru)
                # logging.info('weight: %s: estimated value: %s; true value: %s' % (weight, str(est), str(tru)))

        new_weights = self.update_weights(weights, num_records)

        return [np.dot(ests_list, new_weights),
                np.dot(trus_list, weights),
                np.dot(num_records, weights),
                np.sum(trus_list, axis=1) / self.users.n]

    def sum_handler(self, agg_column, node_indexs):
        k_gran = self.users.ranges[agg_column]['gran']
        k_start = self.users.ranges[agg_column]['min']

        weights = []
        ests = []
        trus = []
        num_records = []
        for k, v in self.users.kv_map[agg_column].items():
            # logging.info('works on k=%s, v=%s' % (str(k), str(v)))

            weight = (k * k_gran + k_start)
            weights.append(weight)
            [perturbed_shard, sensitive_shard, fetched_index] = self.fetch_records(agg_column, k)
            num_records.append(len(perturbed_shard))
            # logging.info('weight: %s, fetched %s records' % (str(weight), str(len(perturbed_shard))))
            if weight == 0 or len(perturbed_shard) == 0:
                ests.append(0)
                trus.append(0)
                continue

            all_results, all_configs, all_multiples = self.aggregator.estimate_all(perturbed_shard, fetched_index)
            all_results, all_configs, all_multiples = self.aggregator.consist(all_results, all_configs, all_multiples)
            est = self.aggregator.estimate_node(all_results, all_configs, all_multiples, node_indexs)
            # logging.info('estimated value: %s' % (str(est)))
            ests.append(est)

            tru = self.calc_nodes(sensitive_shard, node_indexs)
            # logging.info('true value: %s' % (str(tru)))
            trus.append(tru)

        new_weights = self.update_weights(weights, num_records)

        true_result = np.dot(trus, weights)
        result = np.dot(ests, new_weights)

        return [result, true_result]

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
            mask = np.logical_and(sensitive_shard[:, scol_indexs[0]] >= val_indexs[0][0],
                                  sensitive_shard[:, scol_indexs[0]] < val_indexs[0][1])
            for i in range(1, len(scol_indexs)):
                new_mask = np.logical_and(sensitive_shard[:, scol_indexs[i]] >= val_indexs[i][0],
                                          sensitive_shard[:, scol_indexs[i]] < val_indexs[i][1])
                mask = np.logical_and(mask, new_mask)
            result += np.sum(mask)
        return result

    def get_node_from_tree(self, tree, node_index):
        (col_indexs, val_indexs) = node_index
        d = len(col_indexs)
        scol_indexs = [self.users.colind_scolind_map[x] for x in col_indexs]
        return tree[d][tuple(scol_indexs)][tuple(val_indexs)]
