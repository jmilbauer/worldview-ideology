from pathlib import Path
from collections import Counter
import sys
from tqdm import tqdm
import json

corpus = Path(sys.argv[1])
outfile = Path(sys.argv[2])

counts = {}
fs = list(corpus.glob("*.txt"))
print(fs)
for f in fs:
    name = f.stem
    counts[name] = {}
    
    with open(f) as fp:
        for line in tqdm(fp):
            hist = Counter(line.split())
            for k in hist:
                if k not in counts[name]:
                    counts[name][k] = 0
                counts[name][k] += hist[k]
                
json.dump(counts, open(outfile, 'w'))