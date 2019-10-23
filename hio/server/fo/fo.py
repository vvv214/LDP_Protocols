#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18
    
"""

import abc
import math
import numpy as np
from scipy.stats import norm


class FO(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args
        self.ee = 0.0
        self.estimates = []
        self.total_count = 0
        self.p = 0
        self.q = 0
        self.g = 0
        self.var = 0.0
        self.d = 0
        self.c = 0

    @abc.abstractmethod
    def init_e(self, eps, domain):
        return

    def estimate(self, perturbed_datas):
        return

    @abc.abstractmethod
    def perturb(self, datas, domain):
        return

    @abc.abstractmethod
    def support_sr(self, report, value):
        return

    @abc.abstractmethod
    def aggregate(self, domain, perturbed_datas):
        return
