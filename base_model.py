from utilities import *

class BaseModel():
	def __init__(self):
		self.model=None
		self.train_pair_type = 'valid'
		self.test_pair_type = 'valid2'
		self.candidates_type = 'sfml'
		self.knn_candiadates = 18
		self.test_mode = False # To quickly validate the correctness

	def load_golden_res(self, fname):
		res = defaultdict(lambda:False)
		with open(fname,'r') as f:
			for line in f:
				u1,u2 = line.strip().split(',')
				res[(u1,u2)] = True
		return res

	def load_train_pairs(self, pair_type, neg_ratio=1, limit_pos_samples=999999999):
		if self.test_mode:
			limit_pos_samples=1000
		all_pairs = filter_order_list(dictFromFileUnicode(HOME+'candidates/candidate_pairs.{}.{}.lv3.json.gz'.format(pair_type,self.candidates_type)), self.knn_candiadates)\
					+ filter_order_list(dictFromFileUnicode(HOME+'candidates/candidate_pairs.{}.d2v.json.gz'.format(pair_type)), 12)
		pairs = remove_duplicate(all_pairs)
		print ('Train data ({}) recall:'.format(pair_type))
		evaluate(pairs, HOME+'data/original/golden_{}.csv'.format(pair_type))

		golden_res = self.load_golden_res(HOME+'data/original/golden_{}.csv'.format(pair_type))
		pos_pairs = []
		neg_pairs = []
		for pair in pairs:
			u1,u2 = pair[:2]
			if (golden_res[(u1,u2)] or golden_res[(u2,u1)]):
				pos_pairs.append(pair)
			else:
				neg_pairs.append(pair)

		random.shuffle(pos_pairs)
		random.shuffle(neg_pairs)
		pos_pairs = [(_[0],_[1],1.0) for _ in pos_pairs[:limit_pos_samples]]
		neg_pairs = [(_[0],_[1],0.0) for _ in neg_pairs]
		all_pairs = pos_pairs+neg_pairs[:int(len(pos_pairs)*neg_ratio)]
		random.shuffle(all_pairs)

		return all_pairs

	def load_test_pairs(self, limit_pairs=999999999, test_pair_type='valid2'):
		if self.test_mode:
			limit_pairs = 1000
		all_pairs = filter_order_list(dictFromFileUnicode(HOME+'candidates/candidate_pairs.{}.{}.lv3.json.gz'.format(test_pair_type,self.candidates_type)), self.knn_candiadates)\
					+ filter_order_list(dictFromFileUnicode(HOME+'candidates/candidate_pairs.{}.d2v.json.gz'.format(test_pair_type)), 12)
		pairs = remove_duplicate(all_pairs)
		print ('Test data ({}) recall:'.format(test_pair_type))
		evaluate(pairs, HOME+'data/original/golden_{}.csv'.format(test_pair_type))
		random.shuffle(pairs)
		return pairs[:limit_pairs]
