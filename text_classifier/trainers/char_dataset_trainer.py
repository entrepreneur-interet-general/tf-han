import tensorflow as tf

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

    def extract_words(self, document):
        # document = list(sentences = str)
        # Split characters
        out = tf.string_split(document, delimiter=" ")
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=self.hp.pad_word)
        return out

    def extract_sents(self, document_string):
        # Split the document line into sentences
        return tf.string_split([document_string], self.hp.split_doc_token).values

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
