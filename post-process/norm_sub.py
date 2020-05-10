import numpy as np
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
