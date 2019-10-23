#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/28/18
    
"""
import numpy as np

from processor.processor import Processor


class KnowPower(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        domain_size = len(estimates)
        n = sum(self.users.REAL_DIST)
        var = self.fo.var * n

        est_mean = np.mean(estimates)
        # d>2500 is too slow, thus skip
        if domain_size > 2500:
            return estimates

        # it is possible that est_mean < 1
        n_list = np.arange(1, n + 1, dtype=float)
        if est_mean < 1:
            alpha = 1.01
        else:
            alpha = self.search_alpha(est_mean, n_list)

        if self.args.vary == 'top_size':
            ranks = np.argsort(self.users.REAL_DIST)
            high_rand_set = set(ranks[-32:])

        result = np.copy(estimates)
        for i in range(domain_size):
            if self.args.vary == 'top_size' and i not in high_rand_set:
                continue
            n_sublist = n_list[max(0, int(estimates[i]) - 50000): min(n, int(estimates[i]) + 50000)]
            exp_n_list = np.exp(-(estimates[i] - n_sublist) ** 2 / var)
            # the very small ones will be assigned 0, which cannot be raised to any exponent
            exp_n_list[exp_n_list == 0] = 1e-300
            result[i] = np.dot(exp_n_list, (n_sublist ** (1 - alpha))) / np.dot(exp_n_list, (n_sublist ** (0 - alpha)))
        # print(estimates[0:10])
        # print(result[0:10])
        # print(estimates[100:110])
        # print(result[100:110])
        # print(estimates[-10:-1])
        # print(result[-10:-1])
        return result

    @staticmethod
    def power_mean(n_list, alpha):
        return np.sum(n_list ** (1 - alpha)) / np.sum(n_list ** (0 - alpha))

    def search_alpha(self, est_mean, n_list):
        alpha = 2
        while self.power_mean(n_list, alpha) > est_mean:
            alpha *= 2

        r = 0
        h = alpha
        l = alpha / 2
        alpha = (h + l) / 2
        fit_mean = self.power_mean(n_list, alpha)
        while np.fabs(fit_mean - est_mean) > 0.01:
            if fit_mean > est_mean:
                l = alpha
            else:
                h = alpha
            alpha = (h + l) / 2
            fit_mean = self.power_mean(n_list, alpha)
            if r > 1000:
                break
            r += 1
        # if self.args.exp_round == 1:
        #     self.logger.info('%d, %.3f, %.3f, %.3f' % (r, alpha, est_mean, fit_mean))
        return alpha
