#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18

"""
import numpy as np
from server.fo.fo import FO
import xxhash


class LH(FO):

    def init_e(self, eps, domain):
        self.ee = np.exp(eps)
        self.g = int(round(self.ee)) + 1
        self.p = self.ee / (self.ee + self.g - 1)
        self.q = 1 / self.g
        self.var = 4 * self.ee / (self.ee - 1) ** 2

    def perturb(self, datas, domain):
        n = len(datas)
        perturbed_datas = np.zeros(n, dtype=object)
        samples_one = np.random.random_sample(n)
        seeds = np.random.randint(0, n, n)
        for i in range(n):
            y = x = xxhash.xxh32(str(int(datas[i])), seed=seeds[i]).intdigest() % self.g
            # y = x = (datas[i] * seeds[i]) % self.g

            if samples_one[i] > self.p:
                y = np.random.randint(0, self.g - 1)
                if y >= x:
                    y += 1
            perturbed_datas[i] = tuple([y, seeds[i]])
        return perturbed_datas

    def support_sr(self, report, value):
        return report[0] == (xxhash.xxh32(str(value), seed=report[1]).intdigest() % self.g)
        # return report[0] == value * report[1] % self.g

    def aggregate(self, domain, perturbed_datas):
        n = len(perturbed_datas)

        ESTIMATE_DIST = np.zeros(domain, dtype=np.int32)
        for i in range(n):
            for v in range(domain):
                x = xxhash.xxh32(str(v), seed=perturbed_datas[i][1]).intdigest() % self.g
                # x = v * perturbed_datas[i][1] % self.g
                if perturbed_datas[i][0] == x:
                    ESTIMATE_DIST[v] += 1

        return ESTIMATE_DIST
