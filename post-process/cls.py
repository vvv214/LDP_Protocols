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
import cvxpy as cp


class CLS(Processor):
    def calibrate(self, est_dist):
        domain = self.users.show_size
        n = sum(self.users.REAL_DIST)
        obs_dist = self.fo.reverse_normalize(est_dist, n)

        A = np.zeros((domain, domain))
        for i in range(domain):
            for j in range(domain):
                if i == j:
                    A[i][j] = self.fo.p
                else:
                    A[i][j] = self.fo.q

        b = np.copy(obs_dist) / n

        x = cp.Variable(domain)
        # it seems the two gives exactly the same results, but sum_squares is faster
        objective = cp.Minimize(cp.sum_squares(A * x - b))
        # objective = cp.Minimize(cp.norm(A * x - b, 2))
        constraints = [0 <= x, x <= 1, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return x.value * n
