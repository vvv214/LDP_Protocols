import abc
import pickle
from os import path

import numpy as np


class Data(object):
    data = None
    heavy_hitters = None
    heavy_hitter_freq = None

    def __init__(self, n_bits):
        self.n_bits = n_bits
		
		self.heavy_size = 128

		self.heavy_hitters = [0] * self.heavy_size
		self.heavy_hitter_freq = [0] * self.heavy_size
		self.initial_generate()

    def initial_generate(self):
        # each user report one query
        num_bytes = int(self.n_bits / 8)
        f = gzip.open('query_%dbytes_sorted.txt.gz' % num_bytes, 'rb')
        time_total = 657427
        self.data = [0] * time_total
        random_map = np.arange(time_total)
        np.random.shuffle(random_map)
        local_count = 0
        overall_count = 0
        for line in f:
            l = line.split(b'\t')
            query, count = l[0], int(l[1])

            # convert to int
            query_number = 0
            if not len(query) == num_bytes:
                print('wrong', query, count, l, len(query))
                continue
            for i in range(num_bytes):
                query_number |= (query[i] << (8 * i))
                if query[i] > 128:
                    print('wrong', query, query[i])
                    continue
            for i in range(count):
                self.data[random_map[overall_count]] = query_number
                overall_count += 1
            if local_count < self.heavy_size:
                self.heavy_hitters[local_count] = query_number
                self.heavy_hitter_freq[local_count] = count
                local_count += 1
        print('overall count:', overall_count)

    def suffix_tally(self, low, high, cand_dict, shift_num):
        mask = (1 << shift_num) - 1
        buckets = np.zeros(len(cand_dict), dtype=np.int)
        for i in range(int(low), int(high)):
            cur_value = self.data[i] & mask
            if cur_value in cand_dict:
                buckets[cand_dict[cur_value]] += 1
        return buckets
