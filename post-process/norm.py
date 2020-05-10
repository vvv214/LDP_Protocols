import numpy as np
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
