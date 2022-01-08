# standard utilities
from pathlib import Path
import sys
from tqdm import tqdm
import logging
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize

# modeling
from gensim.models.phrases import Phrases, Phraser

logging.basicConfig(level=logging.INFO)

from utils import get_sentences

def tokenize(line):
    """
    basic tokenization of a line.
    """
    result = [w.lower() for s in sent_tokenize(line) for w in word_tokenize(s)]
    return ' '.join(result)

def get_phrases(sentences, mc, th):
    """
    Gets the phrases from a sentence iterator
    
    sentesnce: the sentence iterator.
    mc: min_count (see gensim Phrases)
    th: threshold (see gensim Phrases)
    
    return: gensim Phrases.
    """
    
    return Phrases(sentences, min_count=mc, threshold=th)    

def get_phraser(sentences, mc, th):
    """
    Returns a phraser for the the phrases found in a sentence iterator.
    
    sentences: the sentence iterator
    mc: min_count (see gensim Phrases)
    th: threshold (see gensim Phrases)
    
    return: gensim Phraser
    """
    
    phrases = get_phrases(sentences, mc, th)
    phraser = Phraser(phrases)
    return phraser

def get_multiphraser(fs, params):
    """
    Gets an iterative phraser (ie unigrams -> bigrams -> trigrams)
    
    fs: the files to read from
    params: A list of parameters: [(min_count1, threshold1), (min_count2, threshold2), ...]
    
    return: a function that phrases a tokenized sentence
    """
    
    identity = lambda x: x
    phrasers = []
    def foo(x):
        for p in phrasers:
            x = p[x]
        return x
    for mc, th in params:
        phraser = get_phraser(get_sentences(fs, op=foo, verbose=False), mc=mc, th=th)
        phrasers.append(phraser)   
    return foo
    
def main(sources, targets, phraseseq):
    """
    Reads from source files to build a phraser.
    Then, for each of the source files, phrases and prints to the appropriate target file.
    Also saves the wordcounts across all the source files.
    
    sources: the source files
    targets: the target files to print to. Should be same length as sources
    phraseseq: parameters for the iterative phrasing.
    
    return: None
    """
    
    wordcounts = {}
    
    logging.info("Building multiphraser.")
    multiphraser = get_multiphraser(sources, phraseseq)
    
    logging.info("Phrasing the text")
    for s,t in zip(sources, targets):
        with open(t, 'w') as fp:
            for sent in tqdm(get_sentences([s], op=multiphraser)):
                for w in sent:
                    if w in wordcounts:
                        wordcounts[w] += 1
                    else:
                        wordcounts[w] = 1
                        
                fp.write(f"{' '.join(sent)}\n")
    pickle.dump(wordcounts, open('data/counts.pkl', 'wb'))
        
if __name__ == "__main__":
    """
    USAGE EXAMPLE: python3 processing.py [corpora/raw/] [corpora/prep/]
    """
    
    sourcedir = Path(sys.argv[1]).resolve()
    targetdir = Path(sys.argv[2]).resolve()
    sources = list(sorted(sourcedir.glob('*.txt')))
    targets = [targetdir / s.name for s in sources]
    
    # tweak the phraseseq to go beyond bigrams, change sensitivity, etc.
    # [(5,100)] will do one round, with min_count=5 and threshold=100, building bigrams
    # [(5,100), (5,100)] will do two rounds, with the same parameters. This will produce trigrams and some 4-grams
    phraseseq = [(5,100)]
    main(sources, targets, phraseseq)