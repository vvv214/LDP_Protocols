import math

import numpy as np

import fo


class PEM(object):

    def __init__(self, data, top_k, epsilon):
        self.data = data

        self.k = top_k
		# step_bit_n should be as large as possible (larger step_bit_n incurs higher computational cost)
        self.step_bit_n = 1

    def find(self):
        domain_len = self.args.n_bits
        next_gram_n = self.step_bit_n
        first_gram_n = int(math.log2(self.k)) + 1
        num_iterations = 1 + int(math.ceil((domain_len - first_gram_n) / next_gram_n))

        total_user = len(self.data.data)
        group_users = np.linspace(0, total_user, num_iterations + 1)

        cand_dict = {0: 0}
        for i in range(self.k):
            cand_dict[i] = i

        cand_num = 1 << next_gram_n
        for i in range(num_iterations):
            shift_num = first_gram_n + i * next_gram_n
            prev_shift_num = first_gram_n + (i - 1) * next_gram_n

            new_cand_dict = {}
            count = 0
            for cand, freq in cand_dict.items():
                for suffix in range(cand_num):
                    new_cand_dict[(suffix << prev_shift_num) + cand] = count
                    count += 1

            true_counts = self.data.suffix_tally(group_users[i], group_users[i + 1], new_cand_dict, shift_num)

            true_counts = np.append(true_counts, [int(group_users[1] - sum(true_counts))])
            est_counts = fo.lh(true_counts, self.args.eps)[:-1]
			est_counts = fo.filter_top(est_counts, self.k)

            cand_dict = {}
            for cand, index in new_cand_dict.items():
                if est_counts[index] >= 0:
                    cand_dict[cand] = est_counts[index] * num_iterations
        return cand_dict
