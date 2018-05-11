from pathlib import Path
import pickle
import ujson
import numpy as np
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from time import time


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
    def load_processed(path):
        """Loads a previously saved Processer

        Args:
            path (pathlib.Path or str): path of the pickle file

        Returns:
            Processer: loaded processer without its data attribute
            (to save disk space)
        """
        path = Path(path)
        proc = Processer()
        with (path / 'attributes.json').open('rb') as f:
            attributes = ujson.load(f)

        for attr, val in attributes.items():
            proc.__setattr__(attr, val)

        proc.features = np.loadtxt(
            path / 'features.csv.gz'
        ).reshape(attributes.features_shape)

        proc.labels = np.loadtxt(
            path / 'labels.csv.gz'
        )

        return proc

    def __init__(self):
        """Processer constructor
        """
        self.vocab_size = None
        self.max_sent_len = None
        self.max_doc_len = None
        self.embedding_dim = None
        self.embedding_file = None
        self.data = None
        self.data_path = None
        self.stop_words = [',', '"', "'"]
        self.embedded_data = None
        self.vocab = None
        self.labels = None
        self.features = None
        self.doc_lengths = None
        self.sent_lengths = None
        self.np_embedding_matrix = None
        self.emb_vocab = None
        self.max_items_to_load = None

    def assert_data(self):
        """Checks that the Processer has a data attribute

        Raises:
            ValueError: if the Processer does not have data
        """
        if not self.data:
            raise ValueError('No data in the Processer')

    def load_data(self, data_path, max_items_to_load=1e6):
        """Can load different types of data:
        * toy: to experiment
        * yelp: benchmark

        Others could be implemented.
        Sets the Processer's data attribute as a list of documents

        Args:
            data_path (str or pathlib.Path): where to find the data
        """
        self.data = []
        self.labels = []
        self.data_path = data_path
        self.max_items_to_load = int(max_items_to_load)
        if max_items_to_load == -1:
            max_items_to_load = 1e20
        path = Path().resolve()
        # If path is relative and contains ../ :
        for _ in range(len(data_path.split('../')) - 1):
            path = path.parent
        path /= data_path.split('../')[-1]

        if 'toy' in str(data_path).split('/'):
            for fp in path.iterdir():
                if not fp.is_dir() and fp.suffix == '.txt':
                    with fp.open('r') as f:
                        self.data += f.readlines()
                if not fp.is_dir() and fp.suffix == '.pkl':
                    with fp.open('rb') as f:
                        self.labels = pickle.load(f)
            self.data = [
                l
                for l in ''.join(self.data).split('\n\n')
                if len(l) > 1
            ]
        if 'yelp' in str(data_path).split('/'):
            with path.open('rb') as f:
                for i, line in enumerate(f):
                    if i == max_items_to_load:
                        print('Reached max_items_to_load. Stopping')
                        break
                    review = ujson.loads(line)
                    self.data.append(review['text'])
                    self.labels.append(review['label'])
                self.labels = np.eye(5)[self.labels]

    def delete_stop_words(self, language='english'):
        """Lowers words in the data and ignores those in self.stop_words
        """
        self.assert_data()
        stop_words = set(stopwords.words(language))
        self.data = [
            [
                [word for word in sentence if word not in stop_words]
                for sentence in doc
            ] for doc in self.data
        ]

    def split_into_sentences(self, max_sent_len=None):
        """Splits a text into sentences according to the tokens in
        self.end_of_sentences
        """
        max_sent_len = max_sent_len or -1
        self.assert_data()
        self.data = [
            nltk.sent_tokenize(doc, language='english')[:max_sent_len]
            for doc in self.data
        ]

    def tokenize_words(self, language='english'):
        self.data = [
            [nltk.word_tokenize(s, language=language) for s in doc]
            for doc in self.data
        ]

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
        max_vocab_size = int(max_vocab_size)

        if not max_sent_len and not max_doc_len:
            max_doc_len, max_sent_len = self.get_maxs(data)
            self.max_doc_len = max_doc_len
            self.max_sent_len = max_sent_len

        vocab = vocabulary or {
            '<PAD>': 0,
            '<OOV>': 1
        }

        embedded_data = np.zeros((len(data), max_doc_len, max_sent_len))
        if vocabulary:
            set_vocab = set(vocab.keys())
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence):
                        embedded_data[d][s][w] = vocab.get(
                            word, vocab['<OOV>'])
        else:
            print('Creating vocab')
            count = dict(
                Counter(w for d in data for s in d for w in s)
            )
            num_of_words = sum(list(count.values()))
            frequent_words = sorted(
                count, key=count.get, reverse=True
            )[:max_vocab_size]
            for word in frequent_words:
                vocab[word] = len(vocab)

            set_vocab = set()
            print('Embedding')
            _c = 0
            for d, doc in enumerate(data):
                for s, sentence in enumerate(doc):
                    for w, word in enumerate(sentence):
                        embedded_data[d][s][w] = vocab.get(
                            word, vocab['<OOV>'])
                        _c += 1
                print(
                    '{:2}%'.format(
                        _c * 100 // num_of_words),
                    end='\r')

        self.vocab = vocab
        self.vocab_size = len(vocab)
        return embedded_data

    def save_processed(self, save_path):
        """Dumps the Processer without its
        data attribute to save disk.
        To same from a Trainer: trainer.train_proc.save_processed(
            trainer.hp.dir / 'train_proc'
        )

        Args:
            save_path (str or pathlib.Path): path to the 
            directory where to save the processer
        """
        stime = time()
        # If path is relative and contains ../ :
        save_path = str(save_path)
        if '../' in save_path:
            path = Path()
            for _ in range(len(save_path.split('../')) - 1):
                path = path.parent
            path /= save_path.split('../')[-1]
        elif './' in save_path:
            path = Path() / save_path.split('./')[-1]
        else:
            path = Path(save_path)
        path = path.resolve()

        if not path.is_dir():
            print('Path should be a directory')

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            path / "features.csv.gz",
            self.features.reshape(
                (-1, self.features.shape[-1])
            ).astype(int),
            delimiter=","
        )
        np.savetxt(
            path / "labels.csv.gz",
            self.labels.astype(int),
            delimiter=","
        )
        with(path / "vocab.pkl").open('wb') as f:
            pickle.dump(self.vocab, f)

        attributes = {
            k: self.__getattribute__(k) for k in dir(self)
            if isinstance(self.__getattribute__(k), int) or
            isinstance(self.__getattribute__(k), float) or
            isinstance(self.__getattribute__(k), str)
        }

        attributes['features_shape'] = list(self.features.shape)

        with(path / "attributes.json").open('w') as f:
            ujson.dump(attributes, f)

        print('OK %d s' % int(time() - stime))

    def process(self,
                data_path='../data/toy',
                save_path='',
                vocabulary=None,
                max_sent_len=None,
                max_doc_len=None,
                max_vocab_size=1e6,
                max_items_to_load=-1,
                language='english'):
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
        stime = time()
        print('Loading Data..', end='')
        self.load_data(data_path)

        if 'toy' in str(self.data_path).split('/'):
            print('OK %d. Splitting sentences..' % int(time() - stime), end='')
            self.split_into_sentences(max_sent_len)

            print('OK %d. Tokenizing..' % int(time() - stime), end='')
            self.tokenize_words(language)

        print('OK %d. Deleting stop words..' % int(time() - stime), end='')
        self.delete_stop_words(language)

        print('OK %d. Embedding data..' % int(time() - stime))
        self.embedded_data = self.embed(
            self.data,
            vocabulary=vocabulary,
            max_vocab_size=max_vocab_size,
            max_sent_len=max_sent_len,
            max_doc_len=max_doc_len
        )

        print('OK %d.' % int(time() - stime), end='')

        self.doc_lengths, self.sent_lengths = self.get_lengths(self.data)
        self.features = np.array(self.embedded_data)
        if save_path:
            print('Saving Processer..', end='')
            self.save_processed(save_path)
            print('OK.', end='')
        print()
