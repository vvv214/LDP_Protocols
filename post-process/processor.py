#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/18/18
    
"""

import abc
import math
import numpy as np
from scipy.stats import norm
import logging


class Processor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, users, fo):
        self.logger = logging.getLogger('Processor')
        self.args = args
        self.users = users
        self.fo = fo

    @abc.abstractmethod
    def calibrate(self, est_dist):
        return
