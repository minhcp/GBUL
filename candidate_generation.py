''' 
	+ Build SF-ML vector space model
	+ Generated 100 nearest neighbors for each device-log. 
'''

from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import itertools, os.path
from collections import defaultdict
import cPickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool as Pool
from utilities import *
import random
random.seed(123456)

import argparse
parser = argparse.ArgumentParser(description='parsing arguments')
parser.add_argument('-nthread', action="store",  dest="NTHREAD", type=int)
args = parser.parse_args()
NTHREAD = args.NTHREAD

def sf_ml(fdoc, fgolden, tfidf):
	mpf = defaultdict(int)
	df = defaultdict(int)

	doc_tks = {}
	print 'Counting df {}'.format(fdoc)
	with open(fdoc,'r') as f:
		for line in tqdm(f):
			line = line.strip().split('\t')
			if len(line)==2:
				uid, doc_str = line
				doc_tks[uid] = set(doc_str.split())
				for tk in doc_tks[uid]:
					df[tk]+=1

	print 'Counting mpf'
	with open(fgolden,'r') as f:
		for line in tqdm(f):
			line = line.strip().split(',')
			if len(line)==2:
				u1,u2 = line
				m_urls = doc_tks[u1]&doc_tks[u2]
				for url in m_urls:
					mpf[url]+=1

	new_idf = list(tfidf.idf_)
	n = float(len(tfidf.vocabulary_))
	n_u = len(doc_tks)
	for tk in tfidf.vocabulary_.keys():
		idx = tfidf.vocabulary_[tk]
		n_pairs_h = float((df[tk]*(df[tk]-1))/2)
		match_pairs_h = float(mpf[tk])
		ratio = (n_u) * (match_pairs_h+1.0)/(n_pairs_h+1.0) 
		new_idf[idx] = np.log(ratio) + 1.0
	n_features = len(tfidf.vocabulary_)
	tfidf._tfidf._idf_diag = sp.spdiags(np.array(new_idf), diags=0, m=n_features, n=n_features, format='csr')
	return tfidf

knn = tf = users = timer = n_neighbors = None
def get_predict(line):
	res = []
	user_id, tokens = line.strip().split('\t')
	tmp = knn.kneighbors(X=tf.transform([tokens]), n_neighbors=n_neighbors, return_distance=True)

	for i in range(len(tmp[0][0])):
		if user_id!=users[tmp[1][0][i]]:
				res.append((user_id, users[tmp[1][0][i]], tmp[0][0][i]))
	res = sorted(res, key=lambda x: x[2])
	return  [(r[0],r[1],r[2],i+1) for i,r in enumerate(res)]

def init_for_generate_candidates(knn_k,doc_f):
	global knn, tf, users, n_neighbors
	n_neighbors = knn_k
	tf = pickle.load(open("tmp/tf.pkl", "rb" ))
	row_text = []
	users = []
	with open(doc_f,'r') as f:
		for line in f:
			user, t = line.strip().split('\t')
			users.append(user)
			row_text.append(t)
			
	tf_test = tf.transform(row_text)
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(tf_test, range(1, tf_test.shape[0] + 1))

def generate_candidates(n_neighbors,fmodel_type,fdoc_type, is_sectional=False, write_to_file = True, path_lv = 3):
	''' build suppervised tf-idf like scheme in fmodel_type and derive the representation for fdoc_type
	'''
	model_train_f = 'tmp/facts_url_{}.lv{}{}.txt'.format(fmodel_type,path_lv,'.sectional' if is_sectional else'')
	doc_f = 'tmp/facts_url_{}.lv{}{}.txt'.format(fdoc_type,path_lv,'.sectional' if is_sectional else'')

	if not os.path.isfile(model_train_f):
		create_dump_facts('./data/facts_{}.txt'.format(fmodel_type),model_train_f, lv_limit=path_lv)
	if not os.path.isfile(doc_f):
		create_dump_facts('./data/facts_{}.txt'.format(fdoc_type),doc_f, lv_limit=path_lv)

	vocab = get_vocab(HOME+model_train_f)|get_vocab(HOME+doc_f)
	# tf = TfidfVectorizer(min_df=2, lowercase=False, preprocessor=None, binary=False, sublinear_tf=True).fit(map(lambda x: x.strip().split('\t')[-1], open(model_train_f).readlines()))
	tf = TfidfVectorizer(vocabulary=vocab, lowercase=False, preprocessor=None, binary=False, sublinear_tf=True).fit(map(lambda x: x.strip().split('\t')[-1], open(model_train_f).readlines()))
	tf = sf_ml(model_train_f,'./data/original/golden_{}.csv'.format(fmodel_type),tf)
	print "Size Vocab = {}".format(len(tf.idf_))
	with open("tmp/tf.pkl", 'wb') as handle:
			pickle.dump(tf, handle)

	with open(doc_f,'r') as f:
		datas = f.readlines()

	pool = Pool(NTHREAD,init_for_generate_candidates, (n_neighbors,doc_f,))
	results = [_ for _ in tqdm(pool.imap_unordered(get_predict, datas))]
	data = [y for x in results for y in x]
	data = sorted(data, key=lambda x: x[2])
	if write_to_file:
		dictToFile(data,'candidates/candidate_pairs.{}.{}.lv{}.json.gz'.format(fdoc_type,'sfml',path_lv))
	return data

if __name__ == '__main__':
	for fdoc_type in ['valid','valid2']:
		for path_lv in range(3):
			print '=============Candidate generation ({}-lv{})=============='.format(fdoc_type, path_lv+1)
			all_pairs = generate_candidates(n_neighbors=100, fmodel_type='train', fdoc_type=fdoc_type, is_sectional=True, write_to_file=True, path_lv=path_lv+1)
			all_pairs = filter_order_list(all_pairs, 18)
			evaluate(all_pairs, HOME+'data/original/golden_{}.csv'.format(fdoc_type))