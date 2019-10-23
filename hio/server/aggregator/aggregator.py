import abc
import math
import numpy as np
from scipy.stats import norm
from server.fo.fo_factory import FOFactory


class Aggregator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        self.args = args
        self.fo = FOFactory.create_fo(args.fo, args)

    @abc.abstractmethod
    def estimate_tree(self, perturbed_datas, domain_sizes):
        return

    @abc.abstractmethod
    def perturb(self, datas, domain_sizes):
        return

    def consist(self, hd_tree):
        return hd_tree

    # filter and get highest
    def filter_size(self, thresh_size):
        sorted_indices = np.argsort(self.estimates)
        estimates = np.zeros(len(self.estimates), dtype=np.int)
        for i in sorted_indices[-thresh_size:]:
            estimates[i] = self.estimates[i]
            if estimates[i] <= 0:
                estimates[i] = 1
        self.estimates = estimates
        return self.estimates

    # use a threshold determined by #users, #domains, and variance
    def filter_thres(self, threshold=0):
        if threshold == 0:
            threshold = 1.0 * norm.ppf(1 - self.args.alpha / self.domain_size, 0, 1) * math.sqrt(self.total_count * self.fo.var)
            # threshold = 0.5 * math.sqrt(self.total_count * self.var)
            # threshold = 4.0 * math.sqrt(self.total_count * self.var)
        for i in range(len(self.estimates)):
            if self.estimates[i] < threshold:
                self.estimates[i] = 0
        return

