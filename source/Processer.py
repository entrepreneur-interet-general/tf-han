from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict


class Processer(object):

    @staticmethod
    def build_word_vector_matrix(vector_file, n_words):
        n_words = int(n_words)
        with open(vector_file, 'r') as f:
            print('Loading file...', end='\r')
            lines = f.readlines()
            print('{:20}'.format(' '))
        vectors = np.zeros((
            min([len(lines), n_words]) if n_words else len(lines),
            len(lines[0].split()[1:])
        ))
        words = {}
        for i, l in enumerate(lines):
            if i == n_words:
                print('Reached max_words: stopping here (%d).' % i)
                break
            print('Embedding %d' % i, end='\r')
            values = l.split()
            try:
                vectors[i, :] = np.array(values[1:]).astype(np.float32)
                words[values[0]] = i
            except ValueError:
                print('Ignored vector: ', values[:10], '...')

        return words, vectors

    @staticmethod
    def get_maxs(data):
        """Computes the longest sentence and the longest doc
        for data as a list of documents which are lists of sententences
        which are lists of words

        Args:
            object (list(list(list(str)))): list of documents which are lists
            of sententences
        which are lists of words

        Returns:
            tuple(int): max number of sentences in a doc,
            max number of words in a sentence
        """
        max_doc_len = max(
            len(d) for d in data
        )
        max_sent_len = max(
            len(s) for d in data for s in d
        )
        return max_doc_len, max_sent_len

    @staticmethod
    def get_lengths(data):
        """Get lengths to feed to a dynamic_rnn

        Args:
            data (list(list(list(str)))): data as lsit of docs
            as lists of sentences as lists of words

        Returns:
            tuple(list): list of document lengths, list of sentence
            lengths per document
        """
        doc_lengths = [len(d) for d in data]
        sent_lengths = [[len(s) for s in d] for d in data]
        return doc_lengths, sent_lengths

    @staticmethod
    def load(path):
        """Loads a previously saved Processer

        Args:
            path (pathlib.Path or str): path of the pickle file

        Returns:
            Processer: loaded processer without its data attribute
            (to save disk space)
        """
        path = Path(path)
        with path.open('rb') as f:
            print(
                'Remember, a loaded Processer does not have a data attribute'
            )
            dic = pickle.load(f)
        proc = Processer()
        for k, v in dic.items():
            proc.__setattr__(k, v)
        return proc

    def __init__(self):
        """Processer constructor
        """
        self.vocab_size = None
        self.max_sent_len = None
        self.max_doc_len = None
        self.embedding_dim = None
        self.embedding_file = None
        self.max_words = None
        self.data = None
        self.stop_words = [',', '"', "'"]
        self.end_of_sentences = ['.', '!', '?']
        self.embedded_data = None
        self.vocab = None
        self.labels = None
        self.features = None
        self.doc_lengths = None
        self.sent_lengths = None
        self.np_embedding_matrix = None
        self.emb_vocab = None

    def assert_data(self):
        """Checks that the Processer has a data attribute

        Raises:
            ValueError: if the Processer does not have data
        """
        if not self.data:
            raise ValueError('No data in the Processer')

    def load_data(self, data_path):
        """Can load different types of data:
        * toy: to experiment

        Others could be implemented.
        Sets the Processer's data attribute as a list of documents

        Args:
            data_path (str or pathlib.Path): where to find the data
        """
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
                if not fp.is_dir() and fp.suffix == '.pkl':
                    with fp.open('rb') as f:
                        self.labels = pickle.load(f)
            self.data = [l for l in ''.join(data).split('\n\n') if len(l) > 1]

    def delete_stop_words(self):
        """Lowers words in the data and ignores those in self.stop_words
        """
        self.assert_data()
        temp = self.data
        for w in self.stop_words:
            temp = [' '.join(l.replace(w, ' ').split()).lower() for l in temp]
        self.data = temp

    def split_sentences(self):
        """Splits a text into sentences according to the tokens in
        self.end_of_sentences
        """
        self.assert_data()
        for token in self.end_of_sentences:
            self.data = [s.replace(token, ' %s@@@' % token) for s in self.data]
        self.data = [[' '.join(w.split())
                      for w in s.split('@@@') if w] for s in self.data]

    def embed(self, data, vocabulary=None, max_vocab_size=1e6,
              max_sent_len=None,
              max_doc_len=None):
        """Embeds self.data according to vocabulary. If none is provided,
        it is created on the fly and added to self.vocab

        Sets self.embdedded_data as a numpy array of shape:
        number of docs * max_doc_len * max_sent_len

        Sentences are padded with 0s and Documents are padded with sentences
        full of zeros [0] * max_sent_len

        Args:
            data (list(list(list(str)))): list of documents as lists
            of sentences as lists of words
            vocabulary (dict, optional): Defaults to None. The vocabulary to
            embed the sentences from as {word: index}.
            If None, a new one is created.
            max_vocab_size (number, optional): Defaults to 1e6.
            max length of the vocabulary
            max_sent_len (int, optional): Defaults to None. Max allowed
            length of sentences. If None, it's computed from data
            max_doc_len (int, optional): Defaults to None. idem for documents

        Returns:
            [type]: [description]
        """
        self.assert_data()

        if not max_sent_len and not max_doc_len:
            max_doc_len, max_sent_len = self.get_maxs(data)
            self.max_doc_len = max_doc_len
            self.max_sent_len = max_sent_len

        vocab = vocabulary or defaultdict(int, {
            '<PAD>': 0,
            '<OOV>': 1
        })

        embedded_data = [
            [
                [0 for w in range(max_sent_len)]
                for s in range(max_doc_len)
            ]
            for d in self.data
        ]
        if vocabulary:
            set_vocab = set(vocab.keys())
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence.split()):
                        embedded_data[d][s][w] = vocab[word] \
                            if word in set_vocab else vocab['<OOV>']
        else:
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence.split()):
                        vocab[w] += 1
            frequent_words = sorted(
                vocab, key=vocab.get, reverse=True
            )[:max_vocab_size]
            vocab = {
                k: vocab[k] for k in frequent_words
            }
            set_vocab = set(vocab.keys())

            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence.split()):
                        if word in set_vocab:
                            embedded_data[d][s][w] = vocab[word]
                        else:
                            embedded_data[d][s][w] = vocab['<OOV>']
        self.vocab = vocab
        self.vocab_size = len(vocab)
        return embedded_data

    def save(self, save_path):
        """Dumps the Processer without its
        data attribute to save disk

        Args:
            save_path (str or pathlib.Path): [description]
        """
        path = Path(save_path).resolve()
        if path.is_dir():
            path /= 'processer.pkl'
        dic = {
            k: self.__getattribute__(k)
            for k in dir(self)
            if '__' not in k and
            'data' not in k and
            not callable(self.__getattribute__(k))
        }
        with path.open('wb') as f:
            pickle.dump(dic, f)

    def process(self,
                data_path='../data/toy',
                save_path='',
                vocabulary=None,
                max_sent_len=None,
                max_doc_len=None,
                max_vocab_size=1e6):
        """Main funciton executing the major processing
        steps for the data:
        * load the data
        * delete stop words
        * split text into sentences
        * embed resulting data
        * creates a features attribute from it

            data_path (str, optional): Defaults to '../data/toy'.
                path where to find the data
            save_path (str, optional): Defaults to ''.
                path where to save the processer
            vocabulary (dict, optional): Defaults to None.
                vocabulary to embed the data
            max_sent_len (int, optional): Defaults to None. see embed
            max_doc_len (int, optional): Defaults to None. see embed
            max_vocab_size (number, optional): Defaults to 1e6. see embed
        """

        self.load_data(data_path)
        self.delete_stop_words()
        self.split_sentences()

        self.embedded_data = self.embed(
            self.data,
            vocabulary=vocabulary,
            max_vocab_size=max_vocab_size,
            max_sent_len=max_sent_len,
            max_doc_len=max_doc_len
        )

        self.doc_lengths, self.sent_lengths = self.get_lengths(self.data)
        self.features = np.array(self.embedded_data)
        if save_path:
            self.save(save_path)
