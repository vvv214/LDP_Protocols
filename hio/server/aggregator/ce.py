#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18
    
"""
import numpy as np
from collections import OrderedDict
import itertools

from server.aggregator.aggregator import Aggregator


class CE(Aggregator):

    @staticmethod
    def perturb(datas, domain_sizes, eps):
        pass

    def perturb(self, datas, domain_sizes):
        self.fo.init_e(self.args.epsilon, [np.prod(domain_sizes)])

        new_datas = np.zeros((len(datas), 1), dtype=np.int)
        for domain in range(len(domain_sizes)):
            factor = int(np.prod(domain_sizes[domain + 1:]))
            new_datas[:, 0] += datas[:, domain] * factor

        return self.fo.perturb(new_datas, [np.prod(domain_sizes)])

    def estimate_tree(self, perturbed_datas, domain_sizes):

        n = len(perturbed_datas)
        p = self.fo.ps[0]
        q = self.fo.qs[0]
        domain = int(np.prod(domain_sizes))
        dimension = len(domain_sizes)

        ESTIMATE_DIST = self.fo.aggregate_ce(domain, perturbed_datas[:, 0])

        # step 3: normalize
        a = 1.0 / (p - q)
        b = n * q / (p - q)
        ESTIMATE_DIST = a * ESTIMATE_DIST - b

        # step 4: get ordereddict from resulting list
        result = OrderedDict()
        result[dimension] = OrderedDict()
        result[dimension][tuple(range(dimension))] = OrderedDict()
        domain_size_set = [range(domain_size) for domain_size in domain_sizes]
        all_possible_digits = itertools.product(*domain_size_set)
        count = 0
        for digits in all_possible_digits:
            result[dimension][tuple(range(dimension))][digits] = ESTIMATE_DIST[count]
            count += 1

        # step 5: estimate for lower-dimensional results
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