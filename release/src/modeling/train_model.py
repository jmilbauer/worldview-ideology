from pathlib import Path
from multiprocessing import cpu_count
import sys
import logging
from gensim.models.word2vec import Word2Vec
import queue

logging.basicConfig(level=logging.INFO)

def get_n_tokens(source_files):
    counter = 0
    for f in source_files:
        with open(f, 'r') as fp:
            for line in fp:
                tokens = line.split()
                counter += len(tokens)
    return counter

def iterfile(f_in):
    with open(f_in) as fp:
        for line in fp:
            yield line.split()
            
class MultiFileIter():
    def __init__(self, files):
        self.files = files
        
    def __iter__(self):
        self.q = queue.LifoQueue()
        for f in self.files:
            self.q.put(f)
            
        self.fh = open(self.q.get(), 'r')
        return self
    
    def __next__(self):
        line = self.fh.readline()
        if line == "":
            self.fh.close()
            if not self.q.empty():
                self.fh = open(self.q.get(), 'r')
                return next(self)
            else:
                raise StopIteration
        else:
            return line.split()
        
class FileIter():
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        self.fh = open(self.filename)
        return self
    
    def __next__(self):
        line = self.fh.readline()
        if line == "":
            raise StopIteration
        else:
            return line.split()
            
def train(f_in, f_out, opts):
    n_tokens = get_n_tokens([f_in])
    n_iterations = max(10,min(20,int(opts['itertokens'] / n_tokens)))
    
    model = Word2Vec()
    model = Word2Vec(sentences=FileIter(f_in),
                    size=opts['dims'],
                    window=opts['window'],
                    workers=opts['n_cpu'],
                    sg=1,
                    hs=0,
                    negative=5,
                    min_count=opts['min_count'],
                    max_final_vocab=opts['vocab'],
                    sample=opts['sample'],
                    iter=n_iterations)
    
    model.save(str(f_out))
    logging.info(f"Completed embedding from: {f_in.stem}")
    
def train_multi(dir_in, f_out, opts):
    f_ins = list(dir_in.glob("*.txt"))
    n_tokens = get_n_tokens(f_ins)
    n_iterations = max(10,min(20,int(opts['itertokens'] / n_tokens)))
    
    model = Word2Vec()
    model = Word2Vec(sentences=MultiFileIter(f_ins),
                    size=opts['dims'],
                    window=opts['window'],
                    workers=opts['n_cpu'],
                    sg=1,
                    hs=0,
                    negative=5,
                    min_count=opts['min_count'],
                    max_final_vocab=opts['vocab'],
                    sample=opts['sample'],
                    iter=n_iterations)
    
    model.save(str(f_out))
    logging.info(f"Completed embedding from: {dir_in.stem}")
    
if __name__ == "__main__":    
    source_file = Path(sys.argv[1])
    target_file = Path(sys.argv[2])
    opts = {}
    
    opts['dims'] = int(sys.argv[3])
    opts['window'] = int(sys.argv[4])
    opts['n_cpu'] = min(cpu_count(), 12)
    opts['vocab'] = 15000
    opts['sample'] = float(sys.argv[5])
    opts['itertokens'] = 6000000000
    opts['min_count']= 100
    
    if len(sys.argv) > 6 and sys.argv[6] == 'multi':
        train_multi(source_file, target_file, opts)
    else:
        train(source_file, target_file, opts)