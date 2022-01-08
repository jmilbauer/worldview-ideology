# standard utilities
import sys
from multiprocessing import cpu_count
import logging
from pathlib import Path

# modeling
from gensim.models.word2vec import Word2Vec

# custom
from utils import FileIter

logging.basicConfig(level=logging.INFO)

opts={'dims':300,
     'window':5,
     'n_cpu':cpu_count(),
     'min_count':10,
     'vocab_size':10000,
     'sample':0.0001,
     'n_iter':50
     }

def main(source, target):    
    """
    Main file that trains and saves the Word2Vec model
    
    source: the source file to read from
    target: the target path to save to.
    
    return: None
    """
    
    model = Word2Vec(sentences=FileIter(source),
                     size=opts['dims'],
                     window=opts['window'],
                     workers=opts['n_cpu'],
                     sg=1,
                     hs=0,
                     negative=5,
                     min_count=opts['min_count'],
                     max_final_vocab=opts['vocab_size'],
                     sample=opts['sample'],
                     iter=opts['n_iter']
                    )
    model.save(str(target))

if __name__ == "__main__":
    """
    USAGE EXAMPLE: python3 train.py [corpora/prep/politics.txt] [models/politics.model];
    """
    
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    logging.info(f"Training word2vec for {source}")
    main(source, target)
    logging.info(f"Saved word2vec to {target}")