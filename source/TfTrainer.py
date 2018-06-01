import tensorflow as tf
from Trainer import Trainer
import numpy as np

from utils import extract_sents, extract_words, one_hot_label


class TfTrainer(Trainer):
    def extract_words(self, token):
        # Split characters
        out = tf.string_split(token, delimiter=" ")
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value="<PAD>")
        return out

    def preprocess_dataset(self, doc_ds, label_ds):
        doc_ds = doc_ds.map(extract_sents, self.hp.num_threads)
        doc_ds = doc_ds.map(extract_words, self.hp.num_threads)
        doc_ds = doc_ds.map(self.words.lookup, self.hp.num_threads)

        label_ds = label_ds.map(one_hot_label, self.hp.num_threads)
        return doc_ds, label_ds

    def make_datasets(self):
        with self.graph.as_default():
            with tf.variable_scope("datasets"):
                padded_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None]))
                padding_values = (np.int64(0), 0)

                with tf.variable_scope("vocab-lookup"):
                    self.words = tf.contrib.lookup.index_table_from_file(
                        self.hp.train_words_file, num_oov_buckets=1
                    )

                with tf.variable_scope("train"):
                    train_doc_ds = tf.data.TextLineDataset(self.hp.train_docs_file)
                    train_labels_ds = tf.data.TextLineDataset(self.hp.train_labels_file)

                    dds, lds = self.preprocess_dataset(train_doc_ds, train_labels_ds)
                    train_ds = tf.data.Dataset.zip((dds, lds))
                    train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
                    train_ds = train_ds.repeat()
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
                    val_ds = val_ds.repeat()
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

    def set_procs(self):
        voc = None
        if self.hp.embedding_file:
            voc, mat = self.train_proc.build_word_vector_matrix(
                self.hp.embedding_file, self.hp.max_words
            )
            self.train_proc.np_embedding_matrix = mat
            voc = {k: v + 2 for k, v in voc.items()}
            voc["<PAD>"] = 0
            voc["<OOV>"] = 1
            self.train_proc.vocab = voc
        else:
            with open(self.hp.train_words_file, "r") as f:
                length = sum(1 for line in f) + 1
                self.hp.set_vocab_size(length)

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

        print("Setting Processers...")
        if self.new_procs:
            self.set_procs()

        self.hp.set_data_values(None, None, self.hp.vocab_size, 300)
        self.model.hp.set_data_values(None, None, self.hp.vocab_size, 300)

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
