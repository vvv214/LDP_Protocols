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


class NormMul(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        estimates[estimates < 0] = 0
        total = sum(estimates)
        n = sum(self.users.REAL_DIST)
        return estimates * n / total
