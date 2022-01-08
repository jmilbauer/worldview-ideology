class FileIter(object):
    """
    Single file iterator.
    """
    
    def __init__(self, file, op=None):
        """
        Initializes the iterator.
        
        file: the file to read from. Should have space-separated tokenized words.
        op: Operation to apply to the tokenized line.
        """
        self.file = file
        self.op = op
        
    def __iter__(self):
        if self.op is None:
            self.op = lambda x: x
        
        self.fh = open(self.file)
        return self
        
    def __next__(self):
        line = self.fh.readline()
        if line == "":
            raise StopIteration
        else:
            return self.op(line.strip().split())

def get_sentences(fs, op=None, verbose=False):
    """
    Reads sentences sequentially from multiple files.
    
    fs: paths to read from. Should have space-separated tokenized words
    op: an operation to apply to the tokenized line.
    verbose: verbose
    
    return: iteration of iterable sentences.
    """
    if op is None:
        op = lambda x: x
    for f in fs:
        with open(f) as fp:
            if verbose:
                fp = tqdm(fp)
            for line in fp:
                yield op(line.strip().split())