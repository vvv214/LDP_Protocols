import numpy as np

from processor.processor import Processor


class NormCut(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        order_index = np.argsort(estimates)
        n = sum(self.users.REAL_DIST)

        total = 0
        for i in range(len(order_index)):
            total += estimates[order_index[- 1 - i]]
            if total > n:
                break

        for j in range(i + 1, len(order_index)):
            estimates[order_index[- 1 - j]] = 0

        return estimates * n / total
