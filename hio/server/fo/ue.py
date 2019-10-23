#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18

"""
import numpy as np
from server.fo.fo import FO


class UE(FO):

    def init_e(self, eps, domain):
        self.ee = np.exp(eps)
        self.p = 0.5
        self.q = 1 / (self.ee + 1)
        self.var = 4 * self.ee / (self.ee - 1) ** 2

    def perturb(self, datas, domain):
        n = len(datas)
        perturbed_datas = np.zeros(n, dtype=object)
        samples_one = np.random.random_sample(n)
        for i in range(n):
            samples_zero = np.random.random_sample(domain)

            # enlarge domain to avoid overflow during aggregation
            y = np.zeros(domain, dtype=np.int32)

            for k in range(domain):
                if samples_zero[k] < self.q:
                    y[k] = 1

            v = datas[i]
            y[v] = 1 if samples_one[i] < self.p else 0

            perturbed_datas[i] = y

        return perturbed_datas

    def support_sr(self, report, value):
        return report[value] == 1

    def aggregate(self, domain, perturbed_datas):
        ESTIMATE_DIST = np.sum(perturbed_datas, axis=0)
        return ESTIMATE_DIST
