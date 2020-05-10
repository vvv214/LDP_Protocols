import numpy as np
from processor.processor import Processor


class MLEApx(Processor):
    def calibrate_grr(self, est_dist):
        n = sum(self.users.REAL_DIST)
        estimates = np.copy(est_dist) / n

        obs_dist = self.fo.reverse_normalize(est_dist, n)

        p, q = self.fo.p, self.fo.q

        while min(estimates) < 0:
            pos_map = estimates > 0
            pos_count = sum(pos_map)
            pos_sum = sum(obs_dist[pos_map])
            estimates = (obs_dist * 1.0 / pos_sum * (pos_count * q + (p - q)) - q) / (p - q)
            estimates[~pos_map] = 0
        return estimates * n

    def calibrate(self, est_dist):
        n = sum(self.users.REAL_DIST)
        estimates = np.copy(est_dist) / n
        obs_dist = self.fo.reverse_normalize(est_dist, n) / n
        p, q = self.fo.p, self.fo.q

        pos_map = estimates > 0
        pos_count = sum(pos_map)
        pos_sum = sum(obs_dist[pos_map])
        x = (pos_sum - p - (pos_count - 1) * q) / (p * (1 - p) + (pos_count - 1) * q * (1 - q))
        estimates = (obs_dist - q - q * (1 - q) * x) / (p - q + (p * (1 - p) - q * (1 - q)) * x)
        estimates[~pos_map] = 0

        while min(estimates) < 0:
            pos_map = estimates > 0
            pos_count = sum(pos_map)
            pos_sum = sum(obs_dist[pos_map])
            x = (pos_sum - p - (pos_count - 1) * q) / (p * (1 - p) + (pos_count - 1) * q * (1 - q))
            estimates = (obs_dist - q - q * (1 - q) * x) / (p - q + (p * (1 - p) - q * (1 - q)) * x)
            estimates[~pos_map] = 0

        return estimates * n
