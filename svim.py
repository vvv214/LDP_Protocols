import copy
import logging
import math
import itertools
from scipy.stats import norm
import numpy as np


class SVSM():
    def find(self, fixed_threshold=0):
        single_test_user = self.users.get_size() / 2

        keys, values = self.find_singleton(single_test_user)
        cand_dict = {}
        new_cand_inv = []

        for j in xrange(self.args.true_singleton_size):
            cand_dict[keys[j]] = 2 ** j
            new_cand_inv.append(keys[j])

        # multi test
        buckets = self.users.test_multi(single_test_user + 1, self.users.get_size(), cand_dict)

        self.aggregator.ee = math.exp(self.args.epsilon)
        self.aggregator.total_count = self.users.get_size() - single_test_user
        gram_candidates = self.aggregator.estimate(buckets, use_filter=False)

        for i in xrange(len(gram_candidates)):
            for j in xrange(i + 1, len(gram_candidates)):
                if i | j == j and not i == j:
                    gram_candidates[i] += gram_candidates[j]

        # rebuild
        gram_candidates = self.aggregator.filter(self.args.top_k)
        cand_itemsets = {}
        for j in xrange(1, 2 ** len(cand_dict)):
            if gram_candidates[j] > 0:
                cand_set = []
                for i in xrange(len(cand_dict)):
                    if j & (1 << i) > 0:
                        cand_set.append(new_cand_inv[i])
                l = list(cand_set)
                l.sort()
                cand_itemsets[tuple(l)] = gram_candidates[j]
        return cand_itemsets
	
    def find_singleton(self, test_user=0):
		key_result, value_result = self.singleton_random_limit(test_user, test_user / 10, adaptive_fo=True)
		return self.update_tail_with_reporting_set(key_result, value_result)

	def update_tail_with_reporting_set(self, key_result, value_result):
        addi_total_item = 0
        for i in xrange(self.length_limit + 1, len(self.length_distribution_set)):
            addi_total_item += self.length_distribution_set[i] * (i - self.length_limit)
            if self.length_distribution_set[i] == 0:
                break
        total_item = sum(value_result)

        ratio = addi_total_item * 1.0 / total_item
        for i in xrange(len(value_result)):
            value_result[i] *= (1.0 + ratio)
        return key_result, value_result
		
	def singleton_random(self, test_user, thres=False):
        if self.args.first_phase_truth:
            key_result = self.users.heavy_singletons[:self.args.top_k * 2]
            value_result = []
            for k in key_result:
                value_result.append(self.users.singleton_freq[k])
            return key_result, value_result
	
	def find_percentile_set(self, length_percentile, test_user, singleton_list):
        total = sum(self.length_distribution_set)
        current_total = 0
        for i in xrange(1, len(self.length_distribution_set)):
            if i > 30:
                break
            if self.length_distribution_set[i] == 0:
                break
            current_total += self.length_distribution_set[i]
            if current_total * 1.0 / total > length_percentile:
                break
        return i
	
	def test_length_set(self, length_test_user, length_limit, cand_dict):
        if length_test_user == 0:
            length_test_user = self.users.get_size() / 10

        buckets = self.users.test_length_cand(
            self.users.get_size() - length_test_user,
            self.users.get_size(),
            cand_dict,
            length_limit)

        self.aggregator.ee = math.exp(self.args.epsilon)
        self.aggregator.total_count = length_test_user
        gram_candidates = self.aggregator.estimate(buckets, use_filter=True)
        gram_candidates[0] = 0
        return gram_candidates * 1.0 / length_test_user * self.users.get_size()
	
	def set_grr(self, adaptive_fo, new_cand_dict):
        use_grr = False
        if adaptive_fo:
            if len(new_cand_dict) < self.length_limit * math.exp(self.args.epsilon) * (4 * self.length_limit - 1) + 1:
                self.aggregator.ee = self.length_limit * (math.exp(self.args.epsilon) - 1) + 1
                use_grr = True
            else:
                self.aggregator.ee = math.exp(self.args.epsilon)
        return use_grr
		
    def singleton_random_limit(self, test_user, final_test_user=0, thres=False, adaptive_fo=False, first_phase_output_ratio=1.0):
        single_test_user = test_user
        if single_test_user == 0:
            single_test_user = self.users.get_size() / 2

        # step 1: running limit
        phase_one_user = int(single_test_user * self.args.single_random_alloc)
        singleton_list, value_result = self.singleton_random(phase_one_user, thres=thres)

        key_result = {}
        for i in xrange(len(singleton_list)):
            key_result[(singleton_list[i],)] = i

        # step 2: test with the confined set
        if final_test_user == 0:
            self.length_limit = len(singleton_list)
        else:
            self.length_distribution_set = self.test_length_set(test_user, len(singleton_list), singleton_list)
            self.length_limit = self.find_percentile_set(self.args.length_percentile, final_test_user, singleton_list)
            if not self.args.precise_length_limit == 0:
                self.length_limit = self.args.precise_length_limit

        use_grr = self.set_grr(adaptive_fo, key_result)

        buckets = self.users.test_singleton_cand_limit(phase_one_user + 1, single_test_user - final_test_user,
                                                       key_result,
                                                       set(singleton_list), self.length_limit)

        self.aggregator.total_count = single_test_user - phase_one_user - final_test_user
        
		value_estimates = self.aggregator.estimate(buckets, use_filter=False, use_grr=use_grr)[:-1]
        value_estimates *= 1.0 / self.aggregator.total_count * self.users.get_size() * self.length_limit

        return self.build_result(value_estimates, singleton_list)