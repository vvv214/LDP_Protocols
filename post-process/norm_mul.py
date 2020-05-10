import numpy as np
from processor.processor import Processor


class NormMul(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        estimates[estimates < 0] = 0
        total = sum(estimates)
        n = sum(self.users.REAL_DIST)
        return estimates * n / total
