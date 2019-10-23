#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/28/18
    
"""
import math

import numpy as np
from scipy.stats import norm

from processor.processor import Processor


class NonNegT(Processor):
    def calibrate(self, est_dist):
        domain_size = len(est_dist)
        n = sum(self.users.REAL_DIST)
        var = self.fo.var

        threshold = norm.ppf(1 - self.args.alpha / domain_size, 0, 1) * math.sqrt(n * var)
        estimates = np.copy(est_dist)
        estimates[estimates < threshold] = 0
        return estimates
