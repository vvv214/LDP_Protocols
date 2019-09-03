import copy
import math

import numpy as np
from scipy.stats import norm


class PEM():
    def find(self, segment_len):
        
		num_iterations = 1 + int(math.ceil((domain_len - segment_len) * 1.0 / segment_len))
        total_user = self.n
        group_users = [0] * (num_iterations + 1)
		users_in_each_group = total_user / num_iterations
		for i in range(num_iterations):
			group_users[i + 1] = (i + 1) * users_in_each_group
        group_users[num_iterations] = total_user
        cand_num = self.args.top_k
        self.aggregator.total_count = int(group_users[1])
        self.aggregator.ee = math.exp(self.args.epsilon)
		
        # first iteration
        cand_dict = {}
        cand_inv = []
        cand_count = 0
        for j in range(cand_num):
            cand_dict[j] = cand_count
            cand_count += 1
            cand_inv.append(j)

        # rest of iterations
        for i in range(num_iterations):
            shift_num = (i - 1) * segment_len + self.args.first_gram_n
            mask = (1 << shift_num) - 1

            if i == num_iterations - 1 and not self.args.single_test:
                # last iteration
                self.aggregator.total_count = int(last_n)
                cand_num = self.args.top_k
                segment_len = self.args.domain_len - shift_num

            if segment_len > 20:
                raise Exception('may require too much computational power, change the condition if can tolerate')
            else:
                buckets = self.group_subset_comb(group_users[i], group_users[i + 1], mask, shift_num,
                                                       cand_dict, 1 << segment_len)
                cand_dict = {}
                new_cand_count = 0
                new_cand_inv = []

                gram_candidates = self.aggregator.estimate(buckets.ravel(), thresh_size=cand_num)
                for j in range(cand_count):
                    for k in range(1 << segment_len):
                        if gram_candidates[j * (1 << segment_len) + k] > 0:
                            cand = (k << shift_num) | cand_inv[j]
                            if i == num_iterations - 1:
                                cand_dict[cand] = gram_candidates[j * (1 << segment_len) + k] * num_iterations
                            else:
                                cand_dict[cand] = new_cand_count
                            new_cand_count += 1
                            new_cand_inv.append(cand)
            cand_inv = copy.deepcopy(new_cand_inv)
            cand_count = new_cand_count

        return cand_dict
	
    def group_subset_comb(self, low, high, mask, shift_num, cand_dict, cand_num):
        buckets = np.zeros((len(cand_dict), cand_num), dtype=np.int)
        for i in range(int(low), int(high)):
            last_iter_value = self.RAW_X[i] & mask
            if last_iter_value in cand_dict:
                curr_iter_value = (self.RAW_X[i] >> shift_num) & (cand_num - 1)
                buckets[cand_dict[last_iter_value]][curr_iter_value] += 1
        return buckets