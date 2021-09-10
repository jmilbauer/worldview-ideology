from abc import ABC
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity

def align_svd(source, target):
    product = np.matmul(source.transpose(), target)
    U, s, V = np.linalg.svd(product)
    T = np.matmul(U,V)
    return T

def align_lstsq(source, target):
    T = np.linalg.lstsq(source, target, rcond=None)
    return T

def align_cca(source, target):
    N_dims = source.shape[1]
    cca = CCA(n_components=N_dims, max_iter=2000)
    cca.fit(source, target)
    return cca

class Aligner(ABC):
    def __init__(self, method, source, target, w2id, id2w, mtxA, mtxB, trainvoc):
        self.method = method
        self.src = source
        self.tgt = target
        self.w2idA = w2id
        self.id2wB = id2w
        self.mtxA = mtxA
        self.mtxB = mtxB
        self.anchors = trainvoc
        
    def translate_mtx(self, mtx):
        """
        MTX -> MTX
        """
        pass
    
    def encode_input(self, words):
        """
        [STRING] -> MTX
        """
        embs = [self.mtxA[self.w2idA[w], :] for w in words]
        return np.vstack(embs)
    
    def decode_output(self, mtx, k=1):
        """
        MTX -> [[STRING]]
        """
        similarities = cosine_similarity(mtx, self.mtxB)
        most_similar = np.argsort(similarities, axis=1)[:, ::-1]
        topsims = np.sort(similarities, axis=1)[:, ::-1][:, :k]
        res = [[self.id2wB[i] for i in row[:k]] for row in most_similar]
        return res, topsims
    
    def translate_word(self, word, k=1):
        """
        STRING -> STRING
        """
        encoding = self.encode_input([word])
        translated = self.translate_mtx(encoding)
        decoded = self.decode_output(translated, k=k)
        return decoded[0][:k]
    
    def translate_words(self, words, k=1):
        """
        [STRING] -> [STRING]
        """
        encoding = self.encode_input(words)
        translated = self.translate_mtx(encoding)
        decoded, simscores = self.decode_output(translated, k=k)
        return decoded, simscores

class SVDAligner(Aligner):
    def set_params(self, T):
        self.T = T

    def translate_mtx(self, mtx):
        return mtx.dot(self.T)
    
class LSTSQAligner(Aligner):
    def set_params(self, T):
        self.T = T

    def translate_mtx(self, mtx):
        return mtx.dot(self.T)
    
class CCAAligner(Aligner):
    def set_params(self, cca):
        self.cca = cca

    def translate_mtx(self, mtx):
        return mtx
    
    def translate_word(self, word, k=1):
        tmpA = self.mtxA
        tmpB = self.mtxB
        self.mtxA, self.mtxB = self.cca.transform(tmpA, tmpB)
        res = super().translate_word(word, k=k)
        self.mtxA = tmpA
        self.mtxB = tmpB
        return res
    
    def translate_words(self, words, k=1):
        tmpA = self.mtxA
        tmpB = self.mtxB
        self.mtxA, self.mtxB = self.cca.transform(tmpA, tmpB)
        res, simscores = super().translate_words(words, k=k)
        self.mtxA = tmpA
        self.mtxB = tmpB
        return res, simscores
        
def get_svd_aligner(model_a, model_b, shared, anchorlist):
    # get wordmaps
    w2id = {w:i for i,w in enumerate(shared)}
    id2w = {i:w for i,w in enumerate(shared)}
    awords = list(sorted(list(model_a.wv.vocab)))
    bwords = list(sorted(list(model_b.wv.vocab)))
    w2idA = {w:i for i,w in enumerate(awords)}
    id2wA = {i:w for i,w in enumerate(awords)}
    w2idB = {w:i for i,w in enumerate(bwords)}
    id2wB = {i:w for i,w in enumerate(bwords)}
    
    # build the base matrices
    a_mtx = np.vstack([model_a.wv[w] for w in awords])
    b_mtx = np.vstack([model_b.wv[w] for w in bwords])
    
    # get the anchors
    a_anchor = np.vstack([a_mtx[w2idA[w], :] for w in anchorlist])
    b_anchor = np.vstack([b_mtx[w2idB[w], :] for w in anchorlist])
    
    # get the translation matrix
    T = align_svd(a_anchor, b_anchor)
    
    # build and return the aligner
    aligner = SVDAligner('svd', model_a, model_b, w2idA, id2wB, a_mtx, b_mtx, anchorlist)
    aligner.set_params(T)
    return aligner

def get_lstsq_aligner(model_a, model_b, shared, anchorlist):
    # get wordmaps
    w2id = {w:i for i,w in enumerate(shared)}
    id2w = {i:w for i,w in enumerate(shared)}
    awords = list(sorted(list(model_a.wv.vocab)))
    bwords = list(sorted(list(model_b.wv.vocab)))
    w2idA = {w:i for i,w in enumerate(awords)}
    id2wA = {i:w for i,w in enumerate(awords)}
    w2idB = {w:i for i,w in enumerate(bwords)}
    id2wB = {i:w for i,w in enumerate(bwords)}
    
    # build the base matrices
    a_mtx = np.vstack([model_a.wv[w] for w in awords])
    b_mtx = np.vstack([model_b.wv[w] for w in bwords])
    
    # get the anchors
    a_anchor = np.vstack([a_mtx[w2idA[w], :] for w in anchorlist])
    b_anchor = np.vstack([b_mtx[w2idB[w], :] for w in anchorlist])
    
    # get the translation matrix
    T = align_lstsq(a_anchor, b_anchor)
    
    # build and return the aligner
    aligner = LSTSQAligner('svd', model_a, model_b, w2idA, id2wB, a_mtx, b_mtx, anchorlist)
    aligner.set_params(T)
    return aligner

def get_cca_aligner(model_a, model_b, shared, anchorlist):
    # get wordmaps
    w2id = {w:i for i,w in enumerate(shared)}
    id2w = {i:w for i,w in enumerate(shared)}
    awords = list(sorted(list(model_a.wv.vocab)))
    bwords = list(sorted(list(model_b.wv.vocab)))
    w2idA = {w:i for i,w in enumerate(awords)}
    id2wA = {i:w for i,w in enumerate(awords)}
    w2idB = {w:i for i,w in enumerate(bwords)}
    id2wB = {i:w for i,w in enumerate(bwords)}
    
    # build the base matrices
    a_mtx = np.vstack([model_a.wv[w] for w in awords])
    b_mtx = np.vstack([model_b.wv[w] for w in bwords])
    
    # get the anchors
    a_anchor = np.vstack([a_mtx[w2idA[w], :] for w in anchorlist])
    b_anchor = np.vstack([b_mtx[w2idB[w], :] for w in anchorlist])
    
    # compute CCA
    cca = align_cca(a_anchor, b_anchor)
    
    # build and return the aligner
    aligner = CCAAligner('cca', model_a, model_b, w2idA, id2wB, a_mtx, b_mtx, anchorlist)
    aligner.set_params(cca)
    return aligner