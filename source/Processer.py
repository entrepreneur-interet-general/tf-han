from pathlib import Path
import pickle


class Processer(object):
    @staticmethod
    def get_maxs(data):
        max_doc_len = max(
            len(d) for d in data
        )
        max_sent_len = max(
            len(s) for d in data for s in d
        )
        return max_doc_len, max_sent_len

    @staticmethod
    def load(path):
        path = Path(path)
        with path.open('rb') as f:
            print(
                'Remember, a loaded Processer does not have a data attribute'
            )
            return pickle.load(f)

    def __init__(self):
        self.vocab_size = None
        self.max_sent_len = None
        self.max_doc_len = None
        self.embedding_dim = None
        self.data = None
        self.stop_words = [',', '"', "'"]
        self.end_of_sentences = ['.', '!', '?']
        self.embedded_data = None
        self.vocab = None

    def assert_data(self):
        if not self.data:
            raise ValueError('No data in the Processer')

    def load_data(self, data_path):
        data = []
        if 'toy' in str(data_path).split('/'):
            path = Path().resolve()
            for _ in range(len(data_path.split('../')) - 1):
                path = path.parent
            path /= data_path.split('../')[-1]
            for fp in path.iterdir():
                if not fp.is_dir() and fp.suffix == '.txt':
                    with fp.open('r') as f:
                        data += f.readlines()
            self.data = [l for l in ''.join(data).split('\n\n') if len(l) > 1]

    def delete_stop_words(self):
        self.assert_data()
        temp = self.data
        for w in self.stop_words:
            temp = [' '.join(l.replace(w, ' ').split()).lower() for l in temp]
        self.data = temp

    def split_sentences(self):
        self.assert_data()
        for token in self.end_of_sentences:
            self.data = [s.replace(token, ' %s@@@' % token) for s in self.data]
        self.data = [[' '.join(w.split())
                      for w in s.split('@@@') if w] for s in self.data]

    def embed(self, data, vocabulary=None, max_size=1e6, max_sent_len=None):
        self.assert_data()

        if max_sent_len:
            max_doc_len, _ = self.get_maxs(data)
        else:
            max_doc_len, max_sent_len = self.get_maxs(data)
        if not vocabulary:
            self.max_doc_len = max_doc_len
            self.max_sent_len = max_sent_len
        print('Proc', self.max_doc_len, self.max_sent_len)

        vocab = vocabulary or {
            '<PAD>': 0,
            '<OOV>': 1
        }
        set_vocab = set(vocab.keys())
        embedded_data = [
            [
                [0 for w in range(max_sent_len)]
                for s in d
            ]
            for d in self.data
        ]
        if vocabulary:
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence.split()):
                        embedded_data[d][s][w] = vocab[word] \
                            if word in set_vocab else vocab['<OOV>']
        else:
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence.split()):
                        if word in set_vocab:
                            embedded_data[d][s][w] = vocab[word]
                        else:
                            if len(set_vocab) < max_size:
                                set_vocab.add(word)
                                vocab[word] = len(vocab)
                                embedded_data[d][s][w] = vocab[word]
                            else:
                                embedded_data[d][s][w] = vocab['<OOV>']
        self.vocab = vocab
        self.vocab_size = len(vocab)
        return embedded_data

    def save(self, save_path):
        path = Path(save_path).resolve()
        if path.is_dir():
            path /= 'processer.pkl'
        temp = self.data
        del self.data
        with path.open('wb') as f:
            pickle.dump(self, f)
        self.data = temp

    def process(self,
                data_path='../data/toy',
                save_path='',
                vocabulary=None,
                max_sent_len=None,
                max_size=1e6):

        self.load_data(data_path)
        self.delete_stop_words()
        self.split_sentences()
        self.embedded_data = self.embed(
            self.data,
            vocabulary=vocabulary,
            max_size=max_size,
            max_sent_len=max_sent_len
        )
        if save_path:
            self.save(save_path)
