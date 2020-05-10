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


class NonNeg(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        estimates[estimates < 0] = 0
        return estimates
