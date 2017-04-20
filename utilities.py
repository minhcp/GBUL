from tqdm import tqdm
try:
	import cPickle as pickle
except:
	import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import spatial
import os, json, random
random.seed(123456)

HOME = '' #'C:/Users/PCM/Desktop/GBUL/To_Public/'
MIN_TIMESTAMP = 1461340800 #2016, 04, 23
MAX_TIMESTAMP = 1466611200 #2016, 06, 23

def filter_order_list(pairs, topk, reverse=False):
	'''
	filter topk nearest neighbors
	'''
	print ('filter top_{}'.format(topk))
	order={}
	order_ks = set()
	for p in pairs:
		u = p[0]
		if u in order_ks:
			order[u].append(p)
		else:
			order_ks.add(u)
			order[u] = [p]

	res=[]
	for u in order_ks:
		order[u] =  sorted(order[u], key=lambda x: x[2],reverse=reverse)
		order[u] = order[u][:topk]
		res += order[u]
	return sorted(res, key=lambda x: x[2],reverse=reverse)

def read_user_pairs(fin):
	pairs = []
	with open(fin,'r') as f:
		reader = csv.reader(f, delimiter='\t')
		for r in reader:
			pairs.append(r)
	return pairs

def get_vocab(fname, mindf=2):
	counter = defaultdict(int)
	with open(fname,'r') as f:
		for line in f:
			line = line.strip().split('\t')[-1].split()
			for _ in set(line):
				counter[_]+=1
	vocab = []
	for key,c in counter.iteritems():
		if c>=mindf:
			vocab.append(key)
	return set(vocab)

def remove_duplicate(pairs):
	res = []
	pset = set()
	for p in pairs:
		u1,u2 = p[:2]
		if (u1,u2) not in pset and (u2,u1) not in pset:
			pset.add((u1,u2))
			res.append(p)
	return res

def obj2file(obj,path):
	print ("Saving {}".format(path))
	with open(path, 'wb') as f:
		pickle.dump(obj, f,protocol=2)
	print ("Saving finished.")

def file2obj(path):
	print ("Loading {}".format(path))
	with open(path, 'rb') as f:
		obj = pickle.load(f)
	return obj

def evaluate(f_res, f_golden):
	golden_set = []
	with open (f_golden, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			if len(line)==2:
				u1,u2 = line
				golden_set.append((u1,u2))
	golden_set = set(golden_set)

	if type(f_res) is str:
		res_set = []
		with open (f_res, 'r') as f:
			for line in f:
				line = line.strip().split(',')
				if len(line)==2:
					u1,u2 = line
					res_set.append((min(u1,u2),max(u1,u2))) 
		res_set = set(res_set)
	else:
		res_set = set([(min(_[0],_[1]),max(_[0],_[1])) for _ in f_res])

	tp = len(golden_set&res_set)
	p = float(tp)/len(res_set)
	r = float(tp)/len(golden_set)
	f1 = 2*p*r/(p+r)

	print ('len(res_pairs)={}.len(golden_pairs)={}'.format(len(res_set), len(golden_set)))
	print ('{} {} {}'.format(p,r,f1))

SEC_GAP = 30 #30 mins
def fact_lst2sections(facts,sec_gap=SEC_GAP):
	secs = []
	sec=[]
	prev = facts[0][0]
	for fact in facts:
		assert (prev-fact[0]>=0)
		if prev-fact[0]>sec_gap:
			secs.append(sec)
			sec = []
		sec.append(fact)
		prev = fact[0]
	secs.append(sec)
	return secs

def timestamp2hr(timestamp):
	return datetime.fromtimestamp(timestamp*60+MIN_TIMESTAMP).hour

def timestamp2hr_week(timestamp):
	time = datetime.fromtimestamp(timestamp*60+MIN_TIMESTAMP)
	hr = time.hour
	wd = time.weekday()
	return wd*24+hr

def timestamp2hr_date(timestamp):
	time = datetime.fromtimestamp(timestamp*60+MIN_TIMESTAMP)
	hr = time.hour
	day = time.day
	month = time.month
	assert 4<=month<=6
	return hr + (day-1)*24 + (month-4)*24*31

def get_24hr_distribution(uid, u2f):
		count = defaultdict(float)
		for fact in u2f[uid]:
			count[timestamp2hr(fact[0])]+=1.0
		return np.array([count[_] for _ in range(24)])

def get_24hr_weekly_distribution(uid, u2f):
	count = defaultdict(float)
	for fact in u2f[uid]:
		count[timestamp2hr_week(fact[0])]+=1.0
	return np.array([count[_] for _ in range(24*7)])

def get_24hr_allday_distribution(uid, u2f):
	count = defaultdict(float)
	for fact in u2f[uid]:
		count[timestamp2hr_date(fact[0])]+=1.0
	return np.array([count[_] for _ in range(24*31*3)])

def pearson_correlation(x,y):
	return pearsonr(x,y)[0]

def cosine_distance(x,y):
	return spatial.distance.cosine(x, y)

def load_golden_set(fname):
	res = []
	with open(fname,'r') as f:
		for line in f:
			u1,u2 = line.strip().split(',')
			res.append((u1,u2))
	return set(res)

def load_sample_pos_neg_pairs(pair_type, k_knn, limit_pos_samples=999999999):
	all_pairs = dictFromFileUnicode('candidates/candidate_pairs.{}.tfidf.lv3.json.gz'.format(pair_type))
	all_pairs = filter_order_list(all_pairs, k_knn)
	pairs = remove_duplicate(all_pairs)

	print ('Candidate pairs ({}) recall:'.format(pair_type))
	evaluate(pairs, HOME+'data/original/golden_{}.csv'.format(pair_type))

	golden_res = load_golden_set('./data/original/golden_{}.csv'.format(pair_type))
	pos_pairs = []
	neg_pairs = []
	for pair in pairs:
		u1,u2 = pair[:2]
		if (u1,u2) in golden_res or (u2,u1) in golden_res:
			pos_pairs.append(pair)
		else:
			neg_pairs.append(pair)

	random.shuffle(pos_pairs)
	random.shuffle(neg_pairs)
	pos_pairs = [(_[0],_[1]) for _ in pos_pairs[:limit_pos_samples]]
	neg_pairs = [(_[0],_[1]) for _ in neg_pairs[:limit_pos_samples]]
	return pos_pairs, neg_pairs

def get_parent_url(url):
	if len(url)==0:
		return ''
	else:
		return '_'.join(url.split('_')[:-1])

def create_dump_facts(factf_name,fo_name,lv_limit):
	'''
	fact documents
	uid \t [list of url tokens]
	'''
	if fo_name.find('sectional')>-1:
		create_dump_facts_sectional(factf_name,fo_name,lv_limit)
		return
	urls = dictFromFileUnicode(HOME+'data/fid2url.json.gz')
	with open(factf_name,'r') as f:
		fo = open(fo_name,'w')
		print ('Reading {}'.format(factf_name))
		for line in tqdm(f):
			es = line.strip().split()
			fids = es[2::2]
			doc_str = ''
			for fid in fids:
				tks = ['t'+str(_) for _ in urls[fid][:lv_limit]]
				path=''
				for tk in tks:
					if len(path)==0:
						path = tk 
					else:
						path+='_'+tk
					doc_str += path + ' '
			doc_str = doc_str.strip()
			if len(doc_str)>0:
				fo.write('{}\t{}\n'.format(es[0],doc_str))
		fo.close()

def create_dump_facts_sectional(factf_name,fo_name,lv_limit):
	urls = dictFromFileUnicode(HOME+'data/fid2url.json.gz')
	with open(factf_name,'r') as f:
		fo = open(fo_name,'w')
		print ('Reading {}'.format(factf_name))
		for line in tqdm(f):
			es = line.strip().split()
			facts = [(int(es[2*k+1]),es[2*k+2]) for k in range(len(es)/2)]
			secs = fact_lst2sections(facts)
			doc_str=[]
			for sec in secs:
				sec_urls = []
				for fact in sec:
					fid = fact[1]
					tks = ['t'+str(_) for _ in urls[fid][:lv_limit]]
					path=''
					for tk in tks:
						if len(path)==0:
							path = tk 
						else:
							path+='_'+tk
						sec_urls.append(path)
				doc_str += list(set(sec_urls))
			doc_str = ' '.join(doc_str)
			if len(doc_str)>0:
				fo.write('{}\t{}\n'.format(es[0],doc_str))
		fo.close()

def get_personal_domains_prob(ftype,lv_limit = 4):
	mpf = defaultdict(int)
	df = defaultdict(int)
	is_sectional = True
	vocab = []

	fdoc = 'tmp/facts_url_{}.lv{}{}.txt'.format(ftype,lv_limit,'.sectional' if is_sectional else'')
	if not os.path.isfile(fdoc):
		create_dump_facts('./data/facts_{}.txt'.format(ftype),fdoc, lv_limit=lv_limit)

	doc_tks = {}
	print ('Counting df {}'.format(fdoc))
	with open(fdoc,'r') as f: #fact tmp
		for line in tqdm(f):
			line = line.strip().split('\t')
			if len(line)==2:
				uid, doc_str = line
				tks = doc_str.split()
				doc_tks[uid] = set(tks)
				vocab+=tks
				for tk in doc_tks[uid]:
					df[tk]+=1
	vocab = set(vocab)
	print ('len(vocab)={}'.format(len(vocab)))

	print ('Counting mpf')
	fgolden = './data/original/golden_{}.csv'.format(ftype)
	match_pairs = 0
	with open(fgolden,'r') as f:
		for line in tqdm(f):
			line = line.strip().split(',')
			if len(line)==2:
				u1,u2 = line
				m_urls = doc_tks[u1]&doc_tks[u2]
				for url in m_urls:
					mpf[url]+=1
				match_pairs+=1

	p_urls = []
	pr_dm_count = defaultdict(int)
	p_domains = {}
	p_matching = {}
	pr_count = defaultdict(int)
	print ('Detecting personal domain')
	for tk in tqdm(vocab):
		n_pairs_h = float((df[tk]*(df[tk]-1))/2)
		match_pairs_h = float(mpf[tk])
		if n_pairs_h==0:
			sc=0
		else:
			sc =  match_pairs_h/n_pairs_h 
		p_matching[tk] = sc
		pr = get_parent_url(tk)
		pr_count[pr]+=1
		if sc==1.0 and n_pairs_h>0:
			p_urls.append(pr)
			pr_dm_count[pr]+=1
	for _ in pr_count.keys():
		p_domains[_] = float(pr_dm_count[_])/pr_count[_]
	return p_domains, p_matching

def load_u2facts(fname):
	urls = dictFromFileUnicode(HOME+'data/fid2url.json.gz')
	u2f = {}
	with open(fname,'r') as f:
		print ('Reading {}'.format(fname))
		for i,line in enumerate(tqdm(f)):
			es = line.strip().split()
			uid = es[0]
			times = es[1::2] 
			fids = es[2::2]
			facts = []
			for i,fid in enumerate(fids):
				facts.append((int(times[i]),urls[fid]))
			u2f[uid] = facts
	return u2f

def fact2urls(fact,lv_limit=3):
	res=[]
	tks = ['t'+str(_) for _ in fact[1][:lv_limit]]
	path=''
	for tk in tks:
		if len(path)==0:
			path = tk 
		else:
			path+='_'+tk
		res.append(path)
	return res

def facts2urls(facts,lv_limit=3):
	res=[]
	for fact in facts:
		res+=fact2urls(fact,lv_limit)
	return res

def get_user_group(fgolden):
	adv = defaultdict(list)
	with open(fgolden,'r') as f:
		for line in f:
			line = line.strip().split(',')
			if len(line)==2:
				u1,u2 = line
				adv[u1].append(u2)
				adv[u2].append(u1)

	r = defaultdict(lambda:-1)
	gid = 0
	for u1 in adv.keys():
		for u2 in adv[u1]:
			if r[u1]==-1 and r[u2]==-1:
				r[u1] = gid
				r[u2] = gid
				gid+=1
			elif r[u1]==-1 and r[u2]!=-1:
				r[u1] = r[u2]
			elif r[u1]!=-1 and r[u2]==-1:
				r[u2] = r[u1]
			else:
				assert r[u2] == r[u1]

	g2uids = defaultdict(list)
	for u in r.keys():
		g2uids[r[u]].append(u)
	return g2uids

def pick_best_thr(pairs, golden_set, prefix_length = 0):
	lgs = len(golden_set)
	pair = pairs[0]
	prev = pair[2]
	tp = int((min(pair[0],pair[1]),max(pair[0],pair[1])) in golden_set)
	final_set = set([(min(pair[0],pair[1]),max(pair[0],pair[1]))])
	max_f1 = -1
	thr = -1
	topn = -1
	for i,pair in enumerate(pairs[1:]):
		u1,u2 = min(pair[0],pair[1]),max(pair[0],pair[1])
		if (u1,u2) in final_set:
			continue

		if prev!=pair[2]:
			p = float(tp)/len(final_set)
			r = float(tp)/lgs
			f1 = 2*p*r/(p+r) if (p*r>0) else 0
			if f1 > max_f1 and len(final_set)>prefix_length:
				max_f1 = f1
				pp = p
				rr = r
				thr = prev
				topn = len(final_set)
		final_set.add((u1,u2))
		tp+= int((u1,u2) in golden_set)
		prev = pair[2]
	p = float(tp)/len(final_set)
	r = float(tp)/lgs
	f1 = 2*p*r/(p+r) if (p*r>0) else 0
	if f1 > max_f1:
		max_f1 = f1
		pp = p
		rr = r
		thr = prev
		topn = len(final_set)
	return (pp,rr,max_f1),thr, topn

def selection_by_thr(pairs,threshold, reverse):
	final_set = set()
	for i,pair in enumerate(pairs):
		u1,u2 = min(pair[0],pair[1]),max(pair[0],pair[1])
		if (u1,u2) in final_set:
			continue
		if (reverse and pair[2]<threshold) or (not reverse and pair[2]>threshold):
			break
		final_set.add((u1,u2))
	return  list(final_set)

def evaluate_best_case(all_pairs, mname, ftype, write_to_file = True, reverse = False):
	'''	Select the best threshold.
	'''
	fname = HOME+'results/{}.{}.all_pairs.json.gz'.format(mname,ftype)
	if all_pairs!=None:
		if write_to_file:
			dictToFile(all_pairs,fname)
	else:
		all_pairs = dictFromFileUnicode(fname)

	print ('--------Using threshold----------')
	golden_set = load_golden_set(HOME+'data/original/golden_{}.csv'.format(ftype))
	print ('Top-pairs (by best Threshold)')
	(p,r,f1),thr,topn = pick_best_thr(all_pairs, golden_set)
	print ('thr = {}. len(res_pairs)={}. len(golden_pairs)={}'.format(thr, topn, len(golden_set)))
	print ('{} {} {}'.format(p,r,f1))


def evaluate_with_thr(all_pairs, mname, ftype, thresholds, write_to_file = True, reverse = False):
	'''	Report the performance with specific thresholds
	'''
	fname = HOME+'results/{}.{}.all_pairs.json.gz'.format(mname,ftype)
	if all_pairs!=None:
		if write_to_file:
			dictToFile(all_pairs,fname)
	else:
		all_pairs = dictFromFileUnicode(fname)

	print ('-------- {} - {} ----------'.format(mname, ftype))
	f_golden =  HOME+'data/original/golden_{}.csv'.format(ftype)
	print ('Top-pairs (by pre-defined Threshold)')
	pairs = selection_by_thr(all_pairs,thresholds, reverse=reverse)
	evaluate(pairs,f_golden)

def dictToFile(dict,path):
    print ("Writing to {}".format(path))
    try:
        with gzip.open(path, 'w') as f:
            f.write(json.dumps(dict))
    except:
        # in case the file is too big to zip
        with open(path, 'w') as f:
            f.write(json.dumps(dict))

def dictFromFileUnicode(path):
    print ("Loading {}".format(path))
    try:
        with gzip.open(path,'r') as f:
            return json.loads(f.read().decode('utf-8'))
    except:
        # in case not gzip file
        with open(path, 'r') as f:
            return json.loads(f.read().decode('utf-8'))


def merge_files(finames, foname):
	lines=[]
	for fname in finames:
		with open(fname,'r') as f:
			for line in f:
				lines.append(line)
	with open(foname,'w') as f:
		for line in lines:
			f.write(line)