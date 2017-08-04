# Cross Device User Linking
This work is exteneded from the 1st place solution in CIKM Cup 2016 by the same author.
The approach is described in the following short paper:
Cross-Device User Linking: URL, Session, Visisting Time, and Device-log Embedding (SIGIR 2017)


# Prerequisite
The following data files are required and can be downloaded from https://drive.google.com/drive/folders/0B7XZSACQf0KdNXVIUXEyVGlBZnc:
```
./data/original/facts.json
./data/original/golden_train.csv
./data/original/golden_valid.csv
./data/original/golden_valid2.csv # will be used as testing partition
./data/original/titles.csv
./data/original/urls.csv 
```

# Usage
```
mkdir emb
mkdir candidates
mkdir tmp
mkdir results

python preprocess.py
python candidate_generation.py -nthread 40
python train_device_log_emb.py -nthread 40
python xgb.py -nthread 40
python evaluate.py
```

(author: Minh C. Phan phan0050@e.ntu.edu.sg)
