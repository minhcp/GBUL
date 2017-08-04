from utilities import *
from base_model import *
import numpy as np
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity

import argparse
parser = argparse.ArgumentParser(description='parsing arguments')
parser.add_argument('-nthread', action="store",  dest="N_THREAD", type=int)
args = parser.parse_args()
N_THREAD = args.N_THREAD

class OrderList:
	'''
	Store the score and order of each device log in its nearest neighbors
	'''
	def __init__(self, order_files, topk=100):
		nn_pairs_lst = [filter_order_list(dictFromFileUnicode(_), topk) for _ in order_files]
		self.orders = [Order(_) for _ in nn_pairs_lst]

	def get_order_lst_features(self,u,v):
		res=[]
		for _ in self.orders:
			res+=_.get_order_features(u,v)
		return res
			
class Order:
	def __init__(self, nn_pairs):
		self.orders = {}
		self.scores = {}
		self.initialize(nn_pairs)

	def initialize(self, nn_pairs):
		for p in nn_pairs:
			self.orders[(p[0],p[1])] = p[3]
			u,v = min(p[0],p[1]), max(p[0],p[1])
			self.scores[(u,v)] =  p[2]

	def get_order_features(self, u, v):
		try:
			order2 = self.orders[(v,u)]
		except:
			order2 = 101
		try:
			order1 = self.orders[(u,v)]
		except:
			order1 = 101
		uu,vv = min(u,v), max(u,v)
		try:
			score = self.scores[(uu,vv)]
		except:
			score = 3
		return [order1,order2, score]


class XGB_Model(BaseModel):
	def __init__(self, result_type):
		BaseModel.__init__(self)
		self.model = None
		self.result_type = result_type

	def load_data(self):
		u2f = load_u2facts(HOME+'data/facts_all.txt')
		u2v_model = file2obj(HOME+'emb/dl_embedding.d.300.w.10.user2vec.pkl')

		train_pairs = self.load_train_pairs(self.train_pair_type, neg_ratio=3.0)
		cand_lst = 	['candidates/candidate_pairs.{}.{}.lv{}.json.gz'.format(self.train_pair_type,self.candidates_type,lv) for lv in range(1,4)]\
					+ ['candidates/candidate_pairs.{}.d2v.json.gz'.format(self.train_pair_type)]
		order_lst = OrderList(cand_lst, topk=60) # limit to 60 nearest neighbor due to memory limitation
		p_domains, p_matching = get_personal_domains_prob('train', lv_limit=600)
		x,y = self.get_u_vecs(train_pairs,order_lst,u2f,p_domains, p_matching,u2v_model, with_label=True)

		test_pairs = self.load_test_pairs()
		cand_lst = 	['candidates/candidate_pairs.{}.{}.lv{}.json.gz'.format(self.test_pair_type,self.candidates_type,lv)for lv in range(1,4)]\
					+ ['candidates/candidate_pairs.{}.d2v.json.gz'.format(self.test_pair_type)]
		order_lst = OrderList(cand_lst, topk=60)
		xt = self.get_u_vecs(test_pairs,order_lst,u2f,p_domains, p_matching,u2v_model, with_label=False)

		# temporally verify the model performance
		valid_pairs = self.load_train_pairs(self.test_pair_type, neg_ratio=1, limit_pos_samples=10000)
		xv,yv = self.get_u_vecs(valid_pairs, order_lst,u2f,p_domains, p_matching,u2v_model, with_label=True)
		return x,y,test_pairs,xt,xv,yv
	
	def prob_features(self, probs):
		r=[]
		if len(probs)>0:
			p = np.sum(np.log(probs+1.0))
			r += [p,
				p/len(probs),
				len(probs),
				np.min(probs),
				np.max(probs),
				np.average(probs),
				np.prod(probs)]
		else:
			r += [0.0,
				0.0,
				0,
				0.0,
				0.0,
				0.0,
				0.0]
		probs_pos = probs[probs>0]
		if len(probs_pos)>0:
			r.append(np.prod(probs_pos))
		else:
			r.append(-1)
		return r

	def get_prob_features(self, u1, u2, u2f, p_domains, p_matching):
		res=[]
		urls1 = facts2urls(u2f[u1],lv_limit=100)
		urls2 = facts2urls(u2f[u2],lv_limit=100)
		common_urls = set(urls1)&set(urls2)
		lcu = len(common_urls)
		lcpd = 0
		p_personal_cu=[]
		for _ in common_urls:
			pr = get_parent_url(_)
			try:
				p_personal_cu.append(p_domains[pr])
				if p_domains[pr]==1:
					lcpd+=1
			except:
				continue
		res += [lcpd, np.log(lcpd+1.0), lcu, float(lcpd)/lcu if lcu>0 else -1]
		res += self.prob_features(np.array(p_personal_cu))

		p_match_cu=[]
		for _ in common_urls:
			try:
				p_match_cu.append(p_matching[_])
			except:
				continue
		res += self.prob_features(np.array(p_match_cu))
		return res

	def get_d2v_features(self,u1,u2,u2v_model):
		try:
			v1 = u2v_model[u1]
			v2 = u2v_model[u2]
			cos = cosine_similarity([v1],[v2])[0][0]
			return [cos]
		except:
			return [-1]

	def pair2vec(self, args):
		u1,u2,order_lst,u2f,p_domains,p_matching,u2v_model,is_positive = args
		f=[]

		# SF-ML based features
		f+=order_lst.get_order_lst_features(u1,u2)
		
		# Time-related features
		h1 = get_24hr_distribution(u1,u2f)
		h2 = get_24hr_distribution(u2,u2f)
		f.append(pearson_correlation(h1,h2))
		f.append(cosine_distance(h1,h2))
		f.append(pearson_correlation(np.log(h1+1.0),np.log(h2+1.0)))
		f.append(cosine_distance(np.log(h1+1.0),np.log(h2+1.0)))

		h1 = get_24hr_weekly_distribution(u1,u2f)
		h2 = get_24hr_weekly_distribution(u2,u2f)
		f.append(pearson_correlation(h1,h2))
		f.append(cosine_distance(h1,h2))
		f.append(pearson_correlation(np.log(h1+1.0),np.log(h2+1.0)))
		f.append(cosine_distance(np.log(h1+1.0),np.log(h2+1.0)))

		# Probabilistic features
		f+=self.get_prob_features(u1,u2,u2f,p_domains,p_matching)

		# Semantic Embedding features
		f+=self.get_d2v_features(u1,u2,u2v_model)
		return f+[int(is_positive)]

	def get_u_vecs(self, pairs, order_lst, u2f, p_domains, p_matching, u2v_model, with_label):
		print ('Making user vectors')
		datas = [(_[0],_[1],  order_lst, u2f,p_domains, p_matching, u2v_model, _[-1]) for _ in pairs]
		res = [self.pair2vec(data) for data in tqdm(datas)]
		if with_label:
			return [_[:-1] for _ in res], [_[-1] for _ in res]
		else:
			return [_[:-1] for _ in res]

	def run(self):
		# Load data
		try:
			x,y,test_pairs,xt,xv,yv = file2obj('tmp/xy_xgb.pkl')
		except:
			x,y,test_pairs,xt,xv,yv = self.load_data()
			if not self.test_mode:
				obj2file([x,y,test_pairs,xt,xv,yv],'tmp/xy_xgb.pkl')

		# Create model
		self.model = XGBClassifier(
			learning_rate =0.01,
			n_estimators=4000,
			max_depth=5,
			min_child_weight=1,
			subsample=0.9, 
			objective= 'binary:logistic',
			nthread=N_THREAD,
			)

		# Train
		self.model.fit(np.array(x),np.array(y), verbose=True)
		yv_pred = self.model.predict(np.array(xv))
		print ("Accuracy = {}".format(accuracy_score(yv, yv_pred)))
		print ("P[class-1] = {}".format(precision_score(yv, yv_pred, pos_label=1, average='binary')))
		print ("R[class-1] = {}".format(recall_score(yv, yv_pred, pos_label=1, average='binary')))
		print ("F1[class-1] = {}".format(f1_score(yv, yv_pred, pos_label=1, average='binary')))
		print ("Golden_count={}; Predict_count={}".format(sum(yv), sum(yv_pred)))

		# Test
		predicts = self.model.predict_proba(np.array(xt)).tolist()
		res = [(_[0],_[1],predicts[i][1]) for i,_ in enumerate(test_pairs)]
		res = sorted(res, key=lambda x: x[-1], reverse=True)
		evaluate_best_case(res, self.result_type, self.test_pair_type, write_to_file=True, reverse=True)

if __name__ == '__main__':
	m = XGB_Model(result_type='xgb.all_features')
	m.run()