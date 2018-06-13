import numpy as np
import tensorflow as tf

from ..utils.tf_utils import one_hot_label, one_hot_multi_label
from .trainer import Trainer


class DST(Trainer):
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
            label_ds = label_ds.map(one_hot_label, self.hp.num_threads)
        else:
            label_ds = label_ds.map(one_hot_multi_label, self.hp.num_threads)
        return doc_ds, label_ds

    def make_datasets(self):
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                with tf.variable_scope("datasets"):
                    padded_shapes = (
                        tf.TensorShape([None, None]),
                        tf.TensorShape([None]),
                    )
                    padding_values = (np.int64(0), 0)

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
                        train_ds = tf.data.Dataset.zip((dds, lds))
                        train_ds = train_ds.shuffle(
                            10000, reshuffle_each_iteration=True
                        )
                        train_ds = train_ds.padded_batch(
                            self.batch_size_ph, padded_shapes, padding_values
                        )
                        train_ds = train_ds.prefetch(10)
                        self.train_dataset = train_ds

                    with tf.variable_scope("val"):
                        val_doc_ds = tf.data.TextLineDataset(self.hp.val_docs_file)
                        val_labels_ds = tf.data.TextLineDataset(self.hp.val_labels_file)

                        dds, lds = self.preprocess_dataset(val_doc_ds, val_labels_ds)
                        val_ds = tf.data.Dataset.zip((dds, lds))
                        val_ds = val_ds.shuffle(10000, reshuffle_each_iteration=True)
                        val_ds = val_ds.padded_batch(
                            self.batch_size_ph, padded_shapes, padding_values
                        )
                        val_ds = val_ds.prefetch(10)
                        self.val_dataset = val_ds

                    with tf.variable_scope("infer"):
                        self.features_data_ph = tf.placeholder(
                            tf.int64, [None, None, None], "features_data_ph"
                        )
                        self.labels_data_ph = tf.placeholder(
                            tf.int32, [None, self.hp.num_classes], "labels_data_ph"
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
                        self.mode_ph: 1,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs,
                        self.shuffle_val_ph: False,
                    },
                )
                print("Iterator ready to infer")
            elif is_val:
                self.sess.run(
                    self.val_dataset_init_op,
                    feed_dict={
                        self.mode_ph: 1,
                        self.shuffle_val_ph: True,
                        self.batch_size_ph: self.hp.batch_size,
                    },
                )
            else:
                self.train_bs = self.hp.batch_size
                self.sess.run(
                    self.train_dataset_init_op,
                    feed_dict={
                        self.mode_ph: 0,
                        self.batch_size_ph: self.train_bs,
                        self.shuffle_val_ph: True,
                    },
                )


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
