import numpy as np
import tensorflow as tf

from gensim.models.wrappers import FastText

from ..constants import oov_token
from ..utils.tf_utils import one_hot_label, one_hot_multi_label
from .dataset_trainer import DST


class FT_DST(DST):
    def __init__(self, *args, fast_text_model=None, **kwargs):
        super().__init__(*args, **kwargs)
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
                    if word == "<pad>":
                        new_arr[d, s, w] = np.zeros(300)
                    else:
                        try:
                            new_arr[d, s, w] = self.fast_text_model.wv[word]
                        except Exception as e:
                            new_arr[s, w] = np.ones(300) * 0.00001
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
        self.input_tensor = tf.py_func(
            self.fast_text_lookup,
            [self.input_tensor],
            tf.float32,
            stateful=True,
            name=None,
        )

