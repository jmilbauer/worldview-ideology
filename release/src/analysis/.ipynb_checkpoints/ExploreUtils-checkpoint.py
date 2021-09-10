from nltk.corpus import stopwords, wordnet
from tqdm import tqdm

def get_antonyms(vocab):
    antonyms = []
    for w in tqdm(vocab):
        for synset in wordnet.synsets(w):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.append((w, lemma.antonyms()[0].name()))
    antonyms = set(antonyms)
    return antonyms