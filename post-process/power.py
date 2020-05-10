import numpy as np

from processor.processor import Processor


class Power(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        domain_size = len(estimates)
        n = sum(self.users.REAL_DIST)
        var = self.fo.var * n

        # fit power-law
        est_mean = np.mean(estimates)
        n_list = np.arange(1, n + 1, dtype=float)
        if est_mean < 1:
            alpha = 1.01
        else:
            alpha = self.search_alpha(est_mean, n_list)

        gran = 10
        result_represent_dict = {}

        result = np.copy(estimates)
        for i in range(domain_size):
            result_represent = int(round(estimates[i] / gran))
            if result_represent in result_represent_dict:
                result[i] = result_represent_dict[result_represent]
            else:
                estimates[i] = result_represent * gran + 0.5 * gran
                n_sublist = n_list[max(0, int(estimates[i]) - 25000): min(n, int(estimates[i]) + 25000)]
                exp_n_list = np.exp(-((estimates[i] - n_sublist) ** 2) / 2 / var)
                exp_n_list[exp_n_list == 0] = 1e-32
                denom = np.dot(exp_n_list, (n_sublist ** (0 - alpha)))
                result[i] = np.dot(exp_n_list, (n_sublist ** (1 - alpha)) / denom)
                result_represent_dict[result_represent] = result[i]
        return result

    @staticmethod
    def power_mean(n_list, alpha):
        return np.sum(n_list ** (1 - alpha)) / np.sum(n_list ** (0 - alpha))

    def search_alpha(self, est_mean, n_list):

        alpha = 2
        while self.power_mean(n_list, alpha) > est_mean:
            alpha *= 2

        r = 0
        h = alpha
        l = alpha / 2
        alpha = (h + l) / 2
        fit_mean = self.power_mean(n_list, alpha)
        while np.fabs(fit_mean - est_mean) > 0.01:
            if fit_mean > est_mean:
                l = alpha
            else:
                h = alpha
            alpha = (h + l) / 2
            fit_mean = self.power_mean(n_list, alpha)
            if r > 1000:
                break
            r += 1
        return alpha
