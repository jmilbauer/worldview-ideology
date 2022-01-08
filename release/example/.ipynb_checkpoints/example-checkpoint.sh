#!/bin/bash

# process the raw tokenized data into prepared phrases
python3 preprocessing.py corpora/raw/ corpora/prep/;

# train the fist embeddings
python3 train.py corpora/prep/politics.txt models/politics.word2vec.model;

# train the second embeddings
python3 train.py corpora/prep/the_donald.txt models/the_donald.word2vec.model;
