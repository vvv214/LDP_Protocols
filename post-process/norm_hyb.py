#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/28/18
    
"""
import math

import numpy as np
from scipy.stats import norm

from processor.processor import Processor


class NormHyb(Processor):
    def calibrate(self, est_dist):
        domain_size = len(est_dist)
        n = sum(self.users.REAL_DIST)
        var = self.fo.var

        threshold = min(n, norm.ppf(1 - self.args.alpha / domain_size, 0, 1) * math.sqrt(n * var))
        estimates = np.copy(est_dist)
        # if n or epsilon is small, it is possible some estimates are greater than n, which makes the loop below never terminate
        # impossible_map = estimates > n
        # if impossible_map.any():
        #     estimates[impossible_map] = n / sum(impossible_map)
        above_map = estimates > threshold

        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            neg_map = estimates <= 0
            estimates[neg_map] = 0

            # those bigger than 0 but smaller than threshold needs to be changed
            change_map = np.logical_and(~above_map, ~neg_map)
            if not change_map.any():
                break
            total = sum(estimates)
            diff = (n - total) / sum(change_map)
            estimates[change_map] += diff
        return estimates
