import tensorflow as tf
import numpy as np
from .dataset_trainer import DST
from ..utils.tf_utils import one_hot_label, one_hot_multi_label


class CDST(DST):
    def extract_chars(self, sentences):
        # sentences = list( list(word = str) = sentence) = document
        words = tf.reshape(sentences, (-1,))
        out = tf.string_split(words, delimiter="")
        out = tf.sparse_tensor_to_dense(out, default_value=self.hp.pad_word)
        out = tf.reshape(out, (tf.shape(sentences)[0], tf.shape(sentences)[1], -1))
        return out

    def preprocess_dataset(self, doc_ds, label_ds):
        doc_ds = doc_ds.map(self.extract_sents, self.hp.num_threads)
        doc_ds = doc_ds.map(self.extract_words, self.hp.num_threads)
        doc_ds = doc_ds.map(self.extract_chars, self.hp.num_threads)
        doc_ds = doc_ds.map(self.lookup_table.lookup, self.hp.num_threads)
        if self.hp.multilabel:
            label_ds = label_ds.map(one_hot_multi_label, self.hp.num_threads)
        else:
            label_ds = label_ds.map(one_hot_label, self.hp.num_threads)
        return doc_ds, label_ds

    def set_padding_values(self):
        self.padded_shapes = (
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None]),
        )
        if self.hp.dtype == 32:
            self.padding_values = (np.int32(0), np.int32(0))
        else:
            self.padding_values = (np.int64(0), np.int64(0))

    def set_infer_placehoders(self):
        self.features_data_ph = tf.placeholder(
            tf.int64, [None, None, None, None], "features_data_ph"
        )
        if self.hp.dtype == 32:
            self.labels_data_ph = tf.placeholder(
                tf.int32, [None, self.hp.num_classes], "labels_data_ph"
            )
        else:
            self.labels_data_ph = tf.placeholder(
                tf.int64, [None, self.hp.num_classes], "labels_data_ph"
            )
