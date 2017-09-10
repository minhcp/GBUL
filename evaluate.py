'''
Evaluate xgb models

'''

from utilities import * 

VALID_THR_XGB={
			'xgb.all_features':0.66,
			# 'xgb.no_timef':0.61,
			# 'xgb.no_u2v':0.69,
			# 'xgb.org_tfidf': 0.60,
			# 'xgb.no_term_matching': 0.62,
			# 'xgb.no_per_domain':0.62
			}

if __name__ == '__main__':
	for mname in VALID_THR_XGB.keys():
		evaluate_with_thr(None, mname, 'valid2', thresholds = VALID_THR_XGB[mname], write_to_file = False, reverse = True)