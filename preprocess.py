from tqdm import tqdm
import json
from utilities import *

def make_data_files(fi_name, fo_name, facts):
	res_pairs = []
	uids = []
	with open(fi_name,'r') as f:
		for line in f:
			line = [_.strip() for _ in line.strip().split(',')]
			if len(line)!=2:
				continue
			res_pairs.append(line)
			uids += line
	uids = set(uids)

	# format 
	# uid time fid time fid
	fo = open(fo_name,'w')
	for uid in uids:
		f_str = ' '.join([str(__) for _ in facts[uid] for __ in _])
		fo.write('{} {}\n'.format(uid, f_str))
	fo.close()

def create_facts():
	'''
	Original data comes with:
		+ facts.json,titles.csv, urls.csv
		+ golden_train.csv, golden_valid.csv, golden_valid2.csv
	To Create:
		+golden_valids.csv = golden_valid.csv and golden_valid2.csv
		+golden_all.csv = golden_train.csv and golden_valids.csv

	In the paper, we refer the valid as validation set and valid2 as test set.
	'''
	user_logs = {}
	max_fid_len = 2000
	with open('./data/original/facts.json') as f_in:
		print 'Reading facts.json'
		for line in tqdm(f_in):
			j = json.loads(line.strip())
			uid = j.get('uid')
			facts = j.get('facts')
			f_lst = []
			for x in facts:
				timestamp = x['ts']
				if timestamp>9999999999:
					timestamp = timestamp/1000
				if timestamp>9999999999:
					timestamp = timestamp/1000
				if MAX_TIMESTAMP>timestamp>=MIN_TIMESTAMP:
					timestamp = (timestamp-MIN_TIMESTAMP)/60 # off by min_time to save space
				else:
					continue

				fid = x['fid']
				# assume fids is already sorted by timestamp
				f_lst.append((timestamp,fid))

			user_logs[uid] = f_lst

	merge_files(['./data/original/golden_valid.csv','./data/original/golden_valid2.csv'], './data/original/golden_valids.csv')
	merge_files(['./data/original/golden_valids.csv','./data/original/golden_train.csv'], './data/original/golden_all.csv')

	make_data_files('./data/original/golden_train.csv','./data/facts_train.txt', user_logs)
	make_data_files('./data/original/golden_valid.csv','./data/facts_valid.txt', user_logs)
	make_data_files('./data/original/golden_valid2.csv','./data/facts_valid2.txt', user_logs)
	make_data_files('./data/original/golden_valids.csv','./data/facts_valids.txt', user_logs)
	make_data_files('./data/original/golden_all.csv','./data/facts_all.txt', user_logs)

def create_urls():
	def get_url_id(uid, uid2id):
		try:
			return uid2id[uid]
		except:
			uid2id[uid] = len(uid2id)
			return uid2id[uid]

	uid2id = {}
	urls = {}
	with open('./data/original/urls.csv','r') as f:
		print 'Reading urls.csv'
		for line in tqdm(f):
			line = line.strip().split(',')
			if len(line)!=2:
				continue
			fid, url_str = line
			es = url_str.replace('?','/').split('/')
			es = [get_url_id(_, uid2id) for _ in es]
			urls[fid] = es

	dictToFile(urls,'./data/fid2url.json.gz')
	dictToFile(uid2id,'./data/url2id.json.gz')
	
if __name__ == '__main__':
	create_facts()
	create_urls()