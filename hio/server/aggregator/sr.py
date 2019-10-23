#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18
    
"""
import numpy as np
from collections import OrderedDict
import itertools

from server.aggregator.aggregator import Aggregator


class SR(Aggregator):

    def perturb(self, datas, domain_sizes):

        self.fo.init_e(self.args.epsilon / len(domain_sizes), domain_sizes)

        return self.fo.perturb(datas, domain_sizes)

    def estimate_tree(self, perturbed_datas, domain_sizes):
        n = len(perturbed_datas)
        dimension = len(domain_sizes)
        result = OrderedDict()

        for d in range(1, dimension + 1):

            result[d] = OrderedDict()
            attr_sets = itertools.combinations(range(dimension), d)

            for attr_set in attr_sets:
                result[d][attr_set] = OrderedDict()
                domain_size_subset = [range(domain_sizes[x]) for x in attr_set]
                all_possible_digits = itertools.product(*domain_size_subset)

                for digits in all_possible_digits:
                    same_count = 0
                    for i in range(n):
                        same = True
                        for j in range(len(attr_set)):
                            if not self.fo.support_sr(perturbed_datas[i][attr_set[j]], digits[j], i):
                                same = False
                                break
                        same_count += same

                    minus_count = n * np.prod([self.fo.qs[x] for x in attr_set])
                    for d_ in range(1, d):
                        count_d_ = 0
                        chosen_sets_ = itertools.combinations(range(d), d_)
                        for chosen_set_ in chosen_sets_:
                            coeff = np.prod([self.fo.qs[d] for d in attr_set])
                            coeff *= np.prod([(self.fo.ps[attr_set[x]] / self.fo.qs[attr_set[x]] - 1) for x in chosen_set_])

                            attr_set_ = tuple([attr_set[chosen_set_[k]] for k in range(d_)])
                            digits_ = tuple([digits[chosen_set_[k]] for k in range(d_)])

                            count_d_ += result[d_][attr_set_][digits_] * coeff
                        minus_count += count_d_

                    est = (same_count - minus_count) / np.prod([(self.fo.ps[x] - self.fo.qs[x]) for x in attr_set])
                    result[d][attr_set][digits] = est

        return result