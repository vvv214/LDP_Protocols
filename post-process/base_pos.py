import numpy as np
from processor.processor import Processor


class BasePos(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        estimates[estimates < 0] = 0
        return estimates
