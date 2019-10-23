#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18
    
"""
import numpy as np
from server.fo.fo import FO


class RR(FO):

    def init_e(self, eps, domain):
        self.ee = np.exp(eps)
        self.p = self.ee / (self.ee + domain - 1)
        self.q = 1 / (self.ee + domain - 1)
        self.var = self.q * (1 - self.q) / (self.p - self.q) ** 2

    def perturb(self, datas, domain):
        n = len(datas)
        perturbed_datas = np.zeros(n, dtype=np.int)
        for i in range(n):
            y = x = datas[i]
            p_sample = np.random.random_sample()

            if p_sample > self.p:
                y = np.random.randint(0, domain - 1)
                if y >= x:
                    y += 1
            perturbed_datas[i] = y
        return perturbed_datas

    def support_sr(self, report, value):
        return report == value

    def aggregate(self, domain, perturbed_datas):
        ESTIMATE_DIST = np.zeros(domain)
        unique, counts = np.unique(perturbed_datas, return_counts=True)
        for i in range(len(unique)):
            ESTIMATE_DIST[unique[i]] = counts[i]

        return ESTIMATE_DIST