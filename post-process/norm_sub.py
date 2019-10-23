#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 10/28/18
    
"""
import abc
import math
import numpy as np
import logging
import pandas as pd
from processor.processor import Processor


class NormSub(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        n = sum(self.users.REAL_DIST)
        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            estimates[estimates < 0] = 0
            total = sum(estimates)
            mask = estimates > 0
            diff = (n - total) / sum(mask)
            estimates[mask] += diff
        return estimates
