#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/29/18
    
"""
import logging
import math

import numpy as np


class Evaluator(object):

    def __init__(self, args, server):
        logging.basicConfig(format='%(levelname)s:%(asctime)s: - eval - :%(message)s',
                            level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        self.args = args
        self.server = server

    def error_metric(self):
        if self.args.error_metric == 'query':
            return self.evaluate_query_together()
        elif self.args.error_metric == 'total':
            return self.total_error()

    def load_queries(self):
        with open('./data/query/%s_%s.txt' % (self.args.user_type, self.args.perturb_type), 'r') as f:
            queries = []
            for l in f:
                queries.append(l)
            f.close()
        return queries

    # buffer all queries together, and evaluate all, currently used
    def evaluate_query_together(self):
        if self.args.query_random:
            query = self.load_queries()[0]
            queries = []
            for i in range(self.args.query_size):
                queries.append(query)
        else:
            queries = self.load_queries()

        self.server.perturb_data()

        # [ests, trus, total_weight, selectivity] = self.server.process_query_all(queries)
        [sum, sum_true, total_weight, selectivity, avg, avg_true, count,
         count_true] = self.server.process_query_all_new(queries)
        # logging.info('est:')
        # print(ests)
        # logging.info('trus:')
        # print(trus)
        selectivity = [np.asscalar(x) for x in selectivity]
        volumns = self.server.aggregator.get_vol_sizes()

        results = [0] * len(queries)
        results_norm = [0] * len(queries)
        results_relative = [0] * len(queries)
        results_avg = [0] * len(queries)
        results_norm_avg = [0] * len(queries)
        results_relative_avg = [0] * len(queries)
        results_count = [0] * len(queries)
        results_norm_count = [0] * len(queries)
        results_relative_count = [0] * len(queries)
        # print(ests, trus, total_weight)
        for i in range(len(queries)):
            est, tru = np.asscalar(sum[i]), np.asscalar(sum_true[i])
            results[i] = (est - tru) ** 2
            results_norm[i] = math.fabs(est - tru) / total_weight
            results_relative[i] = math.fabs(est - tru) / tru if tru > 0 else 0

            est, tru = np.asscalar(avg[i]), np.asscalar(avg_true[i])
            results_avg[i] = (est - tru) ** 2
            results_norm_avg[i] = math.fabs(est - tru) / total_weight
            results_relative_avg[i] = math.fabs(est - tru) / tru if tru > 0 else 0

            est, tru = np.asscalar(count[i]), np.asscalar(count_true[i])
            results_count[i] = (est - tru) ** 2
            results_norm_count[i] = math.fabs(est - tru) / total_weight
            results_relative_count[i] = math.fabs(est - tru) / tru if tru > 0 else 0
        return tuple([results, results_norm, volumns, selectivity, results_relative,
                      results_avg, results_norm_avg, results_relative_avg,
                      results_count, results_norm_count, results_relative_count,
                      ])

    # evaluate queries one by one
    def evaluate_query(self):
        self.server.perturb_data()
        with open('./data/query/%s_%s.txt' % (self.args.user_type, self.args.perturb_type), 'r') as f:
            queries = []
            for l in f:
                queries.append(l)
            f.close()

        results = np.zeros(len(queries))
        results_norm = np.zeros(len(queries))
        for i in range(len(queries)):
            logging.info(queries[i])
            [est, tru] = self.server.process_query(queries[i])
            if tru == 0:
                results_norm[i] = 0
            else:
                results_norm[i] = (est / tru - 1) ** 2
            results[i] = (est - tru) ** 2
        return tuple([np.mean(results), np.mean(results_norm)])

    # saved for future
    def total_error(self):
        self.server.perturb_data()
        # todo: obtain the two trees
        estimate_dist, real_dist = {}, {}

        errors = {}
        for d in range(1, self.args.k + 1):
            count = 0
            abs_error = 0
            key_attrs = list(estimate_dist[d].keys())
            random_k_index = np.random.choice(len(key_attrs), min(len(key_attrs), self.args.exp_point))

            for k in random_k_index:
                key_attr = key_attrs[k]
                key_cats = list(estimate_dist[d][key_attr].keys())
                random_k2_index = np.random.choice(len(key_cats), int(self.args.exp_point / len(random_k_index)))

                for k2 in random_k2_index:
                    key_cat = key_cats[k2]
                    abs_error += np.abs(real_dist[d][key_attr][key_cat] - estimate_dist[d][key_attr][key_cat]) ** 2
                    count += 1

            errors[d] = abs_error / count
        return errors
