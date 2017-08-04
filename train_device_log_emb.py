''' 
	+ Train device-log  embedding. 
	+ Generated 100 nearest neighbors for each device-log base on cosine similarity. 
'''

import os
from utilities import * 
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
parser = argparse.ArgumentParser(description='parsing arguments')
parser.add_argument('-nthread', action="store",  dest="N_THREAD", type=int)
args = parser.parse_args()
N_THREAD = args.N_THREAD

class LabeledLineSentence(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):
		with open(self.filename,'r') as f:
			for line in f:
				es = line.strip().split('\t')
				try:
					uids, sen = es[0].strip().split(), es[1].strip()
				except:
					print (line)
					raise
				yield TaggedDocument(sen.split(), uids)

	def get_n_docs(self):
		with open(self.filename,'r') as f:
			return len(f.readlines())

def filter_nxt_same_fid(facts):
	if len(facts)<2:
		return facts
	try:
		# case timestamps with fids
		res=[facts[0]]
		for fact in facts[1:]:
			if fact[-1]!=res[-1][-1]:
				res.append(fact)
	except:
		# case scalar list, just fids
		res=[facts[0]]
		for fid in facts[1:]:
			if fid!=res[-1]:
				res.append(fid)
	return res

def create_corpus(ftype, fvalidtype, foname,lv_limit=3):
	gr2uids = get_user_group(HOME+'data/original/golden_{}.csv'.format(ftype))
	uid2gr = {}
	for gr in gr2uids.keys():
		for uid in gr2uids[gr]:
			uid2gr[uid] = gr

	lines = []
	fid2url = dictFromFileUnicode(HOME+'./data/fid2url.json.gz')
	u2f = load_u2facts(HOME+'data/facts_{}.txt'.format(ftype))
	for uid in u2f.keys():
		guids = ' '.join(gr2uids[uid2gr[uid]])
		facts = filter_nxt_same_fid(u2f[uid])
		lines.append('{}\t{}\n'.format(guids, ' '.join(facts2urls(facts,lv_limit=lv_limit))))		

	u2f = load_u2facts(HOME+'data/facts_{}.txt'.format(fvalidtype))
	for uid in u2f.keys():
		facts = filter_nxt_same_fid(u2f[uid])
		lines.append('{}\t{}\n'.format(uid, ' '.join(facts2urls(facts,lv_limit=lv_limit))))		

	with open(HOME+'data/original/titles.csv','r') as f: #!!! can filter titles in train set
		for line in f:
			fid,tks = line.strip().split(',')
			urls = fid2url[fid]
			if len(urls)<=lv_limit:
				path = '_'.join(['t'+str(_) for _ in urls])
				lines.append('{}\t{}\n'.format(path,tks))
	random.shuffle(lines)
	fo = open(foname,'w')
	for line in lines:
		fo.write(line)
	fo.close()

def get_users(goldenf):
	users=[]
	with open(goldenf,'r') as f:
		for line in f:
			users+= line.strip().split(',')
	return set(users)

def export_u2v(model, fname):
	users = get_users(HOME+'data/original/golden_all.csv')
	ec=0
	res={}
	for u in users:
		try:
			res[u] = model.docvecs[u]
		except:
			ec+=1
	print ('{}/{} users do not have d2v emb'.format(ec, len(users)))
	obj2file(res,fname)

def train_d2v(model_name):
	corpus_inp = HOME+'tmp/{}.d2v.corpus.txt'.format(model_name)
	if not os.path.isfile(corpus_inp):
		create_corpus('train','valids',corpus_inp,lv_limit=3)
	sentences = LabeledLineSentence(corpus_inp)
	model = Doc2Vec(alpha=0.025, min_alpha=0.025, 
					size=300, window=10, 
					min_count=2, workers=N_THREAD)  # use fixed learning rate
	model.build_vocab(sentences)
	for epoch in range(5):
		 model.train(sentences)
		 model.alpha -= 0.002  # decrease the learning rate
		 model.min_alpha = model.alpha  # fix the learning rate, no decay
	return model

# Generate candidates
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing.pool import ThreadPool
def generate_candidates(ftype, fmodel_name):
	user2vec = file2obj(fmodel_name)
	user_vector = []
	user_lst = []
	with open(HOME+'data/facts_{}.txt'.format(ftype),'r') as f:
		for line in f:
			es = line.strip().split()
			uid = es[0]
			v = user2vec[uid]
			user_vector.append(v/np.linalg.norm(v))
			user_lst.append(uid)
	
	knn = KNeighborsClassifier(n_neighbors=1, n_jobs = -1)
	knn.fit(X=user_vector, y=range(1, len(user_lst)+1))
	def find_nearest(i):
		uid = user_lst[i]
		cur_vect = user_vector[i]
		tmp = knn.kneighbors(X=[cur_vect], n_neighbors=100, return_distance=True)
		res = []
		for j in range(len(tmp[0].tolist()[0])):
			if uid!=user_lst[tmp[1].tolist()[0][j]]:
				uid2= user_lst[tmp[1].tolist()[0][j]]
				res.append((uid, uid2, tmp[0][0][j], j+1))
		res = sorted(res, key=lambda x: x[2])
		return  [(r[0],r[1],r[2],j+1) for j,r in enumerate(res)]

	pool = ThreadPool(20)
	nn_pairs = [_ for _ in tqdm(pool.imap_unordered(find_nearest, range(len(user_lst))))]

	nn_pairs = set([y for x in nn_pairs for y in x])
	nn_pairs = sorted(list(nn_pairs), key=lambda x: x[2])
	dictToFile(nn_pairs,HOME+'candidates/candidate_pairs.{}.d2v.json.gz'.format(ftype))
	return nn_pairs


if __name__ == '__main__':
	model_name = 'dl_embedding'
	fmodel_name = 'emb/{}.d.300.w.10.user2vec.pkl'.format(model_name)
	model = train_d2v(model_name)
	export_u2v(model, fmodel_name)

	all_pairs = generate_candidates('valid', fmodel_name)
	# all_pairs = filter_order_list(all_pairs, 18)
	evaluate(all_pairs, HOME+'data/original/golden_{}.csv'.format('valid'))
	
	all_pairs = generate_candidates('valid2', fmodel_name)
	# all_pairs = filter_order_list(all_pairs, 18)
	evaluate(all_pairs, HOME+'data/original/golden_{}.csv'.format('valid2'))