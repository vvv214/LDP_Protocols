import heapq
import math

import numpy as np

import fo


class SVSM():
    def __init__(self, data, top_k, epsilon):
        self.data = data
        self.top_k = top_k
        self.epsilon = epsilon

    def find(self):
        n = len(self.data.data)
        single_test_user = int(0.4 * n)

        key_list, est_freq = self.find_singleton(single_test_user)
        cand_itemsets = self.find_itemset(key_list, est_freq, single_test_user, n)

        return cand_itemsets

    def find_singleton(self, single_test_user):
        phase1_user = int(single_test_user * 0.4)
        phase2_user = phase1_user + int(0.1 * single_test_user)
        phase3_user = single_test_user

        # step 1: find singleton candidate set
        true_singleton_dist = self.data.test_single(0, phase1_user)
        est_singleton_dist = fo.lh(true_singleton_dist, self.epsilon)
        top_singleton = 2 * self.top_k
        singleton_list, value_result = self.build_result(est_singleton_dist, range(len(est_singleton_dist)),
                                                         top_singleton)

        # step 2: find an appropriate length
        key_result = {}
        for i in range(len(singleton_list)):
            key_result[(singleton_list[i],)] = i

        length_percentile = 0.9
        length_distribution = self.test_length_singleton(phase1_user + 1, phase2_user, len(singleton_list),
                                                         singleton_list)
        length_limit = self.find_percentile_set(length_distribution, length_percentile)

        # step 3: test with the confined set
        use_grr, eps = self.set_grr(key_result, length_limit)
        true_singleton_dist = self.data.test_singleton_cand_limit(phase2_user + 1, phase3_user, key_result,
                                                                  set(singleton_list), length_limit)

        if use_grr:
            value_estimates = fo.rr(true_singleton_dist, eps)[:-1]
        else:
            value_estimates = fo.lh(true_singleton_dist, eps)[:-1]
        value_estimates *= single_test_user / length_percentile / (phase3_user - phase2_user)

        top_singleton = self.top_k
        key_list, est_freq = self.build_result(value_estimates, singleton_list, top_singleton)
        return key_list, est_freq

    def find_itemset(self, singletons, singleton_freq, single_test_user, multi_test_user):

        # step 1: build itemset candidate
        set_cand_dict, set_cand_dict_inv = self.get_set_cand_thres_prod(singletons, singleton_freq)

        # step 2: itemset size distribution
        length_percentile = 0.9
        percentile_test_user = single_test_user + int(0.2 * (multi_test_user - single_test_user))
        length_distribution_set = self.test_length_itemset(single_test_user + 1, percentile_test_user,
                                                           len(set_cand_dict), set_cand_dict)
        length_limit = self.find_percentile_set(length_distribution_set, length_percentile)

        # step 3: itemset est
        true_itemset_dist = self.data.test_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict,
                                                      length_limit)
        use_grr, eps = self.set_grr(true_itemset_dist, length_limit)

        if use_grr:
            set_freq = fo.rr(true_itemset_dist, eps)[:-1]
        else:
            set_freq = fo.lh(true_itemset_dist, eps)[:-1]
        set_freq *= single_test_user / length_percentile / (multi_test_user - percentile_test_user)

        self.update_tail_with_reporting_set(length_limit, length_distribution_set, set_cand_dict, set_freq)

        return self.build_set_result(singletons, singleton_freq, set_freq, set_cand_dict_inv)

    # ===== auxiliary functions for singletons
    def build_result(self, value_estimates, key_list, top_singleton):
        sorted_indices = np.argsort(value_estimates)
        key_result = []
        value_result = []
        for j in sorted_indices[-top_singleton:]:
            key_result.append(key_list[j])
            value_result.append(value_estimates[j])

        return key_result, value_result

    def find_percentile_set(self, length_distribution, length_percentile):
        total = sum(length_distribution)
        current_total = 0
        for i in range(1, len(length_distribution)):
            if i > 30:
                break
            current_total += length_distribution[i]
            if current_total / total > length_percentile:
                break
        return i

    def set_grr(self, new_cand_dict, length_limit):
        eps = self.epsilon
        use_grr = False
        if len(new_cand_dict) < length_limit * math.exp(self.epsilon) * (4 * length_limit - 1) + 1:
            eps = math.log(length_limit * (math.exp(self.epsilon) - 1) + 1)
            use_grr = True
        return use_grr, eps

    def test_length_singleton(self, user_start, user_end, length_limit, cand_dict):
        true_length_dist = self.data.test_length_cand(user_start, user_end, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, self.epsilon)
        return est_length_dist

    # ===== auxiliary functions for itemset: constructing candidate set
    def get_set_cand_thres_prod(self, key_list, est_freq):

        cand_dict = {}
        for i in range(len(key_list)):
            cand_dict[key_list[i]] = est_freq[i]

        normalized_values = np.zeros(len(est_freq))
        for i in range(len(est_freq)):
            normalized_values[i] = (est_freq[i] * 0.9 / est_freq[-1])
        cand_dict = {}
        cand_dict_prob = {}
        new_cand_inv = []
        self.build_tuple_cand_bfs(cand_dict_prob, cand_dict, new_cand_inv, key_list, normalized_values)
        cand_list = list(cand_dict.keys())
        cand_value = list(cand_dict_prob.values())
        sorted_indices = np.argsort(cand_value)
        new_cand_dict = {}
        new_cand_inv = []
        for j in sorted_indices[-self.top_k:]:
            new_cand_dict[cand_list[j]] = len(new_cand_inv)
            new_cand_inv.append(tuple(cand_list[j]))
        return new_cand_dict, new_cand_inv

    def build_tuple_cand_bfs(self, cand_dict_prob, cand_dict, new_cand_inv, keys, values):
        ret = []
        cur = []
        for i in range(len(keys)):
            heapq.heappush(ret, (values[i], tuple([i])))
            heapq.heappush(cur, (-values[i], tuple([i])))
        while len(cur) > 0:
            new_cur = []
            while len(cur) > 0:
                (prob, t) = heapq.heappop(cur)
                for j in range(t[-1] + 1, len(keys)):
                    if -prob * values[j] > ret[0][0]:
                        if len(ret) >= 3 * len(keys):
                            heapq.heappop(ret)
                        l = list(t)
                        l.append(j)
                        heapq.heappush(ret, (-prob * values[j], tuple(l)))
                        heapq.heappush(new_cur, (prob * values[j], tuple(l)))
            cur = new_cur

        while len(ret) > 0:
            (prob, t) = heapq.heappop(ret)
            if len(t) == 1:
                continue
            l = list(t)
            new_l = []
            for i in l:
                new_l.append(keys[i])
            new_t = tuple(new_l)
            cand_dict[new_t] = len(new_cand_inv)
            cand_dict_prob[new_t] = prob
            new_cand_inv.append(new_t)

    def test_length_itemset(self, user_start, user_end, length_limit, cand_dict):
        true_length_dist = self.data.test_length_itemset(user_start, user_end, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, self.epsilon)
        return est_length_dist

    def update_tail_with_reporting_set(self, length_limit, length_distribution_set, key_result, value_result):
        addi_total_item = 0
        for i in range(length_limit + 1, len(length_distribution_set)):
            addi_total_item += length_distribution_set[i] * (i - length_limit)
            if length_distribution_set[i] == 0:
                break
        total_item = sum(value_result)

        ratio = addi_total_item * 1.0 / total_item
        for i in range(len(value_result)):
            value_result[i] *= (1.0 + ratio)
        return key_result, value_result

    def build_set_result(self, singletons_keys, singleton_freq, set_freq, set_cand_dict_inv):
        current_estimates = np.concatenate((singleton_freq, set_freq), axis=0)
        results = {}
        sorted_indices = np.argsort(current_estimates)
        for j in sorted_indices[-self.top_k:]:
            if j < len(singletons_keys):
                results[tuple([singletons_keys[j]])] = singleton_freq[j]
            else:
                l = list(set_cand_dict_inv[j - len(singletons_keys)])
                l.sort()
                results[tuple(l)] = current_estimates[j]

        return results
