#!/bin/sh

mkdir emb
mkdir candidates
mkdir tmp
mkdir results

python preprocess.py
python candidate_generation.py -nthread 40
python train_device_log_emb.py -nthread 40
python xgb.py -nthread 40
python evaluate.py