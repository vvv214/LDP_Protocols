import pickle
from os import path

import numpy as np


class Data(object):

    def __init__(self, dataname, limit):
        self.data = None
        self.dict_size = 42178
        user_total = 990002
        random_map = np.arange(user_total)
        np.random.shuffle(random_map)
        overall_count = 0
        user_file_name = '%s-data/%s' % (dataname, dataname)
        if not path.exists(user_file_name + '.pkl'):
            data = [0] * user_total
            f = open(user_file_name + '.dat', 'r')
            for line in f:
                if len(line) == 0:
                    break
                if line[0] == '#':
                    continue
                queries = line.split(' ')

                data[random_map[overall_count]] = [0] * len(queries)
                for i in range(len(queries)):
                    query = int(queries[i])
                    data[random_map[overall_count]][i] = query
                data[random_map[overall_count]].sort()
                overall_count += 1
                if overall_count >= user_total:
                    break
            pickle.dump(data, open(user_file_name + '.pkl', 'wb'))
        self.data = pickle.load(open(user_file_name + '.pkl', 'rb'))
        # only use part of the data to get results quickly
        self.data = self.data[:limit]

    # ===== singleton testing methods
    def test_single(self, low, high):
        results = np.zeros(self.dict_size, dtype=np.int)
        for i in range(low, high):
            if len(self.data[i]) == 0:
                continue
            rand_index = np.random.randint(len(self.data[i]))
            value = self.data[i][rand_index]
            results[value] += 1
        return results

    def test_length_cand(self, low, high, cand_list, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        cand_set = set(cand_list)
        for i in range(low, high):
            X = self.data[i]
            # V = cand_set.intersection(X)
            # value = len(V)
            value = 0
            for item in X:
                if item in cand_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    def test_singleton_cand_limit(self, low, high, key_dict, singleton_set, length_limit):
        results = np.zeros(len(singleton_set) + 1, dtype=np.int)
        for i in range(low, high):
            values = []
            x = self.data[i]
            for item in x:
                if item in singleton_set:
                    values.append(item)
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[(values[rand_index],)]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(singleton_set)
                if rand_index < len(values):
                    result = key_dict[(values[rand_index],)]
            results[result] += 1
        return results

    # ===== itemset testing methods
    def test_length_itemset(self, low, high, cand_dict, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        for i in range(low, high):
            current_set = singleton_set.intersection(set(self.data[i]))
            if len(current_set) == 0:
                continue
            value = 0
            for cand in cand_dict:
                if set(cand) <= current_set:
                    value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    def test_cand_limit(self, low, high, cand_dict, length_limit):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.int)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))

        for i in range(low, high):
            current_set = singleton_set.intersection(set(self.data[i]))
            if len(current_set) == 0:
                continue
            subset_count = 0
            subset_indices = []
            for cand in cand_dict:
                if set(cand) <= current_set:
                    subset_count += 1
                    subset_indices.append(cand_dict[cand])

            if subset_count > length_limit:
                rand_index = np.random.randint(subset_count)
                result = subset_indices[rand_index]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(cand_dict)
                if rand_index < subset_count:
                    result = subset_indices[rand_index]

            buckets[result] += 1
        return buckets
