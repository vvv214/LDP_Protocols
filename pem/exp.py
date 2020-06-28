from data import Data
from pem import PEM


def evaluate(cand_dict, k=0):
	# print('# ==== evaluation ====')
	# print('# identified ', len(cand_dict))
	# print('#',)
	# for r in cand_dict:
	#     print('%x,' % r,)
	# print()
	found_count = 0
	found_stat = 0
	for i in range(k):
		if data.heavy_hitters[i] in cand_dict:
			print('# found %x, rank %d, count %d' % (self.users.heavy_hitters[i], i, self.users.heavy_hitter_freq[i]))
			found_count += 1
			found_stat += self.users.heavy_hitter_freq[i]
	print('# identified %d, %d of them are among top %d' % (len(cand_dict), found_count, k))
	return found_count, found_stat
	
	
def main():
    data = Data()
    finder = PEM(data, top_k=32, epsilon=4)
    cand_dict = finder.find()
    print(cand_dict)


main()
