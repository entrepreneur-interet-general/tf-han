import numpy as np
import tensorflow as tf

from ..utils.tf_utils import one_hot_label, one_hot_multi_label
from .trainer import Trainer
from ..constants import oov_token


class DST(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_values = None
        self.vocab = None
        self.reversed_vocab = None

    def extract_words(self, token):
        # Split characters
        out = tf.string_split(token, delimiter=" ")
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=self.hp.pad_word)
        return out

    def extract_sents(self, string):
        # Split the document line into sentences
        return tf.string_split([string], self.hp.split_doc_token).values

    def preprocess_dataset(self, doc_ds, label_ds):
        doc_ds = doc_ds.map(self.extract_sents, self.hp.num_threads)
        doc_ds = doc_ds.map(self.extract_words, self.hp.num_threads)
        doc_ds = doc_ds.map(self.words.lookup, self.hp.num_threads)
        if self.hp.multilabel:
            label_ds = label_ds.map(one_hot_multi_label, self.hp.num_threads)
        else:
            label_ds = label_ds.map(one_hot_label, self.hp.num_threads)
        return doc_ds, label_ds

    def make_datasets(self):
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                with tf.variable_scope("datasets"):
                    padded_shapes = (
                        tf.TensorShape([None, None]),
                        tf.TensorShape([None]),
                    )
                    if self.hp.dtype == 32:
                        self.padding_values = (np.int64(0), np.int32(0))
                    else:
                        self.padding_values = (np.int64(0), np.int64(0))

                    with tf.variable_scope("vocab-lookup"):
                        self.words = tf.contrib.lookup.index_table_from_file(
                            self.hp.train_words_file, num_oov_buckets=1
                        )

                    with tf.variable_scope("train"):
                        train_doc_ds = tf.data.TextLineDataset(self.hp.train_docs_file)
                        train_labels_ds = tf.data.TextLineDataset(
                            self.hp.train_labels_file
                        )

                        dds, lds = self.preprocess_dataset(
                            train_doc_ds, train_labels_ds
                        )
                        self.train_dataset = tf.data.Dataset.zip((dds, lds))
                        if self.hp.restrict > 0:
                            self.train_dataset = self.train_dataset.take(
                                int(self.hp.restrict)
                            )
                        self.train_dataset = self.train_dataset.shuffle(
                            10000, reshuffle_each_iteration=True
                        )
                        self.train_dataset = self.train_dataset.padded_batch(
                            self.batch_size_ph, padded_shapes, self.padding_values
                        )
                        self.train_dataset = self.train_dataset.prefetch(10)

                    with tf.variable_scope("val"):
                        val_doc_ds = tf.data.TextLineDataset(self.hp.val_docs_file)
                        val_labels_ds = tf.data.TextLineDataset(self.hp.val_labels_file)

                        dds, lds = self.preprocess_dataset(val_doc_ds, val_labels_ds)
                        self.val_dataset = tf.data.Dataset.zip((dds, lds))
                        self.val_dataset = self.val_dataset.shuffle(
                            10000, reshuffle_each_iteration=True
                        )
                        self.val_dataset = self.val_dataset.padded_batch(
                            self.batch_size_ph, padded_shapes, self.padding_values
                        )
                        self.val_dataset = self.val_dataset.prefetch(10)

                    with tf.variable_scope("infer"):
                        self.features_data_ph = tf.placeholder(
                            tf.int64, [None, None, None], "features_data_ph"
                        )
                        if self.hp.dtype == 32:
                            self.labels_data_ph = tf.placeholder(
                                tf.int32, [None, self.hp.num_classes], "labels_data_ph"
                            )
                        else:
                            self.labels_data_ph = tf.placeholder(
                                tf.int64, [None, self.hp.num_classes], "labels_data_ph"
                            )
                        infer_dataset = tf.data.Dataset.from_tensor_slices(
                            (self.features_data_ph, self.labels_data_ph)
                        )
                        self.infer_dataset = infer_dataset.batch(self.batch_size_ph)

    def prepare(self):
        """Runs all steps necessary before training can start:
            * Processers should have their data ready
            * Hyperparameter is updated accordingly
            * Datasets are initialized
            * Trainer is built
        """
        if self.prepared:
            print("Already prepared, skipping.")
            return

        with open(self.hp.train_words_file, "r") as f:
            # Need vocabsize to initialize embedding matrix
            length = sum(1 for line in f) + 1
            self.hp.vocab_size = length

        print("Ok. Setting Datasets...")
        self.make_datasets()
        self.make_iterators()
        print("Ok. Building graph...")
        self.build()
        print("Ok. Saving hp...")
        self.hp.dump(to="json")
        print("OK.")
        self.prepared = True

    def initialize_iterators(self, is_val=False, inference_data=None):
        """Initializes the train and validation iterators from
        the Processers' data

            is_val (bool, optional): Defaults to False. Whether to initialize
                the validation (True) or training (False) iterator
        """
        with self.graph.as_default():
            if inference_data is not None:
                feats = inference_data
                labs = np.zeros((len(feats), self.hp.num_classes))
                bs = len(feats)
                self.sess.run(
                    self.infer_dataset_init_op,
                    feed_dict={
                        self.mode_ph: "infer",
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs,
                    },
                )
                print("Iterator ready to infer")
            elif is_val:
                self.sess.run(
                    self.val_dataset_init_op,
                    feed_dict={
                        self.mode_ph: "val",
                        self.batch_size_ph: self.hp.batch_size,
                    },
                )
            else:
                self.train_bs = self.hp.batch_size
                self.sess.run(
                    self.train_dataset_init_op,
                    feed_dict={
                        self.mode_ph: "train",
                        self.batch_size_ph: self.train_bs,
                    },
                )

    def lookup(self, features, vocab=None):
        if not self.vocab:
            if vocab:
                print("Setting vocab")
                self.vocab = vocab
                self.reversed_vocab = {v: k for k, v in vocab.items()}
            else:
                print("Computing vocab")
                self.vocab = {}
                self.reversed_vocab = {}
                with open(self.hp.train_words_file, "r") as f:
                    for i, l in enumerate(f):
                        w = l[:-1]  # strip final \n
                        self.vocab[w] = i
                        self.reversed_vocab[i] = w
                m = max(list(self.reversed_vocab.keys()))
                self.reversed_vocab[m + 1] = oov_token
                self.vocab[oov_token] = m + 1
        data = []
        for doc in features:
            document = []
            for sent in doc:
                sentence = []
                for word_index in sent:
                    if word_index != 0:
                        sentence.append(self.reversed_vocab[word_index])
                if sentence:
                    document.append(sentence)
            data.append(document)
        return data


if __name__ == "__main__":
    from importlib import reload
    import Hyperparameter as hyp
    import Trainer as trainer
    import utils as uts

    reload(hyp)
    reload(trainer)
    reload(uts)

    t = DST("HAN")
    t.train()
