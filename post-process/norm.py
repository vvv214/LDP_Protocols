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


class Norm(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        total = sum(estimates)
        n = sum(self.users.REAL_DIST)
        domain_size = len(est_dist)
        diff = (n - total) / domain_size
        estimates += diff
        return estimates
