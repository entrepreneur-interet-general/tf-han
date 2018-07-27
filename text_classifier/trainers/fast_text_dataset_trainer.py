import numpy as np
import tensorflow as tf

from gensim.models.wrappers import FastText

from ..constants import oov_token
from ..utils.tf_utils import one_hot_label, one_hot_multi_label
from .dataset_trainer import DST
from ..hyperparameters import HP


class FT_DST(DST):
    def __init__(self, *args, hp=None, fast_text_model=None, **kwargs):
        hp = hp or HP()
        hp.fast_text = True
        hp.embedding_dim = 300
        super().__init__(*args, hp=hp, **kwargs)
        self.decode = np.vectorize(lambda x: x.decode("utf-8"))
        assert fast_text_model or self.hp.fast_text_model_file
        self.fast_text_model = fast_text_model or FastText.load_fasttext_format(
            self.hp.fast_text_model_file
        )

    def fast_text_lookup(self, arr):
        decoded_arr = self.decode(arr)
        new_arr = np.zeros((*arr.shape, 300))
        for d, doc in enumerate(decoded_arr):
            for s, sent in enumerate(doc):
                for w, word in enumerate(sent):
                    if word == self.hp.pad_word:
                        new_arr[d, s, w] = np.zeros(300)
                    else:
                        try:
                            new_arr[d, s, w] = self.fast_text_model.wv[word]
                        except KeyError as e:
                            # print(">> Caught ", e)
                            new_arr[d, s, w] = np.ones(300) * 0.00001
        return new_arr.astype(np.float32)

    def preprocess_dataset(self, doc_ds, label_ds):
        doc_ds = doc_ds.map(self.extract_sents, self.hp.num_threads)
        doc_ds = doc_ds.map(self.extract_words, self.hp.num_threads)
        # doc_ds = doc_ds.map(self.words.lookup, self.hp.num_threads)
        if self.hp.multilabel:
            label_ds = label_ds.map(one_hot_multi_label, self.hp.num_threads)
        else:
            label_ds = label_ds.map(one_hot_label, self.hp.num_threads)
        return doc_ds, label_ds

    def set_input_tensors(self):
        super().set_input_tensors()
        self.model._original_input = self.input_tensor
        self.input_tensor = tf.py_func(
            self.fast_text_lookup,
            [self.input_tensor],
            tf.float32,
            stateful=False,
            name=None,
        )

    def set_padding_values(self):
        if self.hp.dtype == 32:
            self.padding_values = (self.hp.pad_word, np.int32(0))
        else:
            self.padding_values = (self.hp.pad_word, np.int64(0))

    def set_infer_placehoders(self):
        self.features_data_ph = tf.placeholder(
            tf.string, [None, None, None], "features_data_ph"
        )
        if self.hp.dtype == 32:
            self.labels_data_ph = tf.placeholder(
                tf.int32, [None, self.hp.num_classes], "labels_data_ph"
            )
        else:
            self.labels_data_ph = tf.placeholder(
                tf.int64, [None, self.hp.num_classes], "labels_data_ph"
            )
