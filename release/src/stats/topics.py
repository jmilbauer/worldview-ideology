from pathlib import Path
import sys
import json

import numpy as np
from sklearn.cluster import KMeans
from gensim.models.word2vec import Word2Vec

import logging

logging.basicConfig(level=logging.INFO)

def build_matrix(modelfile):
    model = Word2Vec.load(str(modelfile))
    vocab = list(sorted(list(model.wv.vocab)))
    mtx = np.vstack([model.wv[w] for w in vocab])
    return mtx, vocab
    
def get_clusters(mtx, wordlist):
    clustering = KMeans(n_clusters=100).fit(mtx)
    res = {}
    for c, w in zip(clustering.labels_, wordlist):
        c = str(c)
        if c not in res:
            res[c] = []
        res[c].append(w)
    logging.info(f"{len(res)} clusters.")
    logging.info(f"{sum(map(len, res.values()))} words.")
    return res

if __name__ == "__main__":
    embedding_file = Path(sys.argv[1])
    outfile = Path(sys.argv[2])
    
    mtx, wordlist = build_matrix(embedding_file)
    logging.info("Built matrix")
    
    clusters = get_clusters(mtx, wordlist)
    logging.info("Computed clusters")
    
    json.dump(clusters, open(outfile, 'w'))
    logging.info("Saved.")
    