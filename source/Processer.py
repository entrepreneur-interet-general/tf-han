from pathlib import Path
import pickle
import numpy as np


class Processer(object):
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
            return pickle.load(f)

    def __init__(self):
        """Processer constructor
        """
        self.vocab_size = None
        self.max_sent_len = None
        self.max_doc_len = None
        self.embedding_dim = None
        self.data = None
        self.stop_words = [',', '"', "'"]
        self.end_of_sentences = ['.', '!', '?']
        self.embedded_data = None
        self.vocab = None
        self.labels = None
        self.features = None

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
        Sets the Processer's data attribute.

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

    def embed(self, data, vocabulary=None, max_size=1e6,
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
            max_size (number, optional): Defaults to 1e6. max length of the
            vocabulary
            max_sent_len (int, optional): Defaults to None. Max allowed
            length of sentences. If None, it's computed from data
            max_doc_len (int, optional): Defaults to None. idem for documents

        Returns:
            [type]: [description]
        """
        self.assert_data()

        if not max_sent_len and not max_doc_len:
            max_doc_len, max_sent_len = self.get_maxs(data)

        if not vocabulary:
            self.max_doc_len = max_doc_len
            self.max_sent_len = max_sent_len

        vocab = vocabulary or {
            '<PAD>': 0,
            '<OOV>': 1
        }
        set_vocab = set(vocab.keys())
        embedded_data = [
            [
                [0 for w in range(max_sent_len)]
                for s in range(max_doc_len)
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
        """Dumps the Processer without its
        data attribute to save disk

        Args:
            save_path (str or pathlib.Path): [description]
        """
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
                max_doc_len=None,
                max_size=1e6):
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
            max_size (number, optional): Defaults to 1e6. see embed
        """

        self.load_data(data_path)
        self.delete_stop_words()
        self.split_sentences()
        self.embedded_data = self.embed(
            self.data,
            vocabulary=vocabulary,
            max_size=max_size,
            max_sent_len=max_sent_len,
            max_doc_len=max_doc_len
        )
        self.features = np.array(self.embedded_data)
        if save_path:
            self.save(save_path)
