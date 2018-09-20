import tensorflow as tf
from tensorflow.python.client import device_lib

from ..utils.tf_utils import bidirectional_rnn, task_specific_attention
from .model import Model
from ..utils.custom_layer import BNLSTMCell

MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell


class HAN(Model):
    def get_rnn_cell(self, is_training_ph):
        if is_training_ph is not None and self.hp.use_bnlstm:
            return BNLSTMCell(self.hp.cell_size, is_training_ph)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config):
            local_device_protos = device_lib.list_local_devices()
            gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
        if gpus:
            return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.hp.cell_size)
        return tf.nn.rnn_cell.GRUCell(self.hp.cell_size)

    def __init__(self, hp, is_training, graph=None):
        super().__init__(hp, graph)
        self.np_embedding_matrix = None
        self.embedding_matrix = None
        self.batch = None
        self.docs = None
        self.sents = None
        self.sent_cell_fw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )
        self.sent_cell_bw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )
        self.doc_cell_fw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )
        self.doc_cell_bw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )
        self.is_training = is_training

        self.val_ph = None
        self.embedded_inputs = None
        self.embedded_sentences = None
        self.sentence_lengths = None
        self.encoded_sentences = None
        self.attended_sentences = None
        self.sentence_output = None
        self.doc_inputs = None
        self.encoded_docs = None
        self.attended_docs = None
        self.doc_output = None
        self.logits = None
        self.prediction = None
        self.embedding_words = None
        self.doc_lengths = None
        self._original_input = None
        self.word_output = None

    def _unstack_lengths(self):
        with tf.variable_scope("unstack-lengths"):
            (self.batch, self.docs, self.sents) = tf.unstack(
                tf.shape(
                    self.input_tensor if not self.hp.fast_text else self._original_input
                )
            )

    def set_lengths(self):
        self._unstack_lengths()
        with tf.variable_scope("set-lengths"):
            if self.hp.fast_text:
                self.doc_lengths = tf.count_nonzero(
                    tf.reduce_sum(tf.reduce_sum(self.input_tensor, axis=-1), axis=-1),
                    axis=-1,
                )
                self.sentence_lengths = tf.count_nonzero(
                    tf.reduce_sum(self.input_tensor, axis=-1), axis=-1
                )
            else:
                self.doc_lengths = tf.count_nonzero(
                    tf.reduce_sum(self.input_tensor, axis=-1), axis=-1
                )
                self.sentence_lengths = tf.count_nonzero(self.input_tensor, axis=-1)

    def set_sentence_level(self):
        with tf.variable_scope("sentence-level"):
            with tf.variable_scope("embedding-lookup"):
                if self.hp.fast_text:
                    self.embedded_inputs = self.input_tensor
                elif self.hp.chan:
                    self.embedded_inputs = self.word_output
                else:
                    self.embedded_inputs = tf.nn.embedding_lookup(
                        self.embedding_matrix, self.input_tensor
                    )

            with tf.variable_scope("reshape"):
                self.embedded_sentences = tf.reshape(
                    self.embedded_inputs,
                    [
                        self.batch * self.docs,
                        self.sents,
                        self.hp.embedding_dim
                        if not self.hp.chan
                        else self.hp.cell_size,
                    ],
                )

                self.sentence_lengths = tf.reshape(
                    self.sentence_lengths, [self.batch * self.docs]
                )

            with tf.variable_scope("bidir-rnn") as scope:
                self.encoded_sentences, _ = bidirectional_rnn(
                    self.sent_cell_fw,
                    self.sent_cell_bw,
                    self.embedded_sentences,
                    self.sentence_lengths,
                    scope=scope,
                )

            with tf.variable_scope("attention") as scope:
                self.attended_sentences = task_specific_attention(
                    self.encoded_sentences, self.hp.cell_size, scope=scope
                )

            with tf.variable_scope("dropout"):
                self.sentence_output = tf.contrib.layers.dropout(
                    self.attended_sentences,
                    keep_prob=self.hp.dropout,
                    is_training=self.is_training,
                )

    def set_doc_level(self):
        with tf.variable_scope("doc-level"):
            with tf.variable_scope("reshape"):
                self.doc_inputs = tf.reshape(
                    self.sentence_output, [self.batch, self.docs, self.hp.cell_size]
                )

            with tf.variable_scope("bidir-rnn") as scope:
                self.encoded_docs, _ = bidirectional_rnn(
                    self.doc_cell_fw,
                    self.doc_cell_bw,
                    self.doc_inputs,
                    self.doc_lengths,
                    scope=scope,
                )

            with tf.variable_scope("attention") as scope:
                self.attended_docs = task_specific_attention(
                    self.encoded_docs, self.hp.cell_size, scope=scope
                )

            with tf.variable_scope("dropout"):
                self.doc_output = tf.contrib.layers.dropout(
                    self.attended_docs,
                    keep_prob=self.hp.dropout,
                    is_training=self.is_training,
                )

    def set_classifier(self):
        with tf.variable_scope("classifier"):
            try:
                activation = getattr(tf.nn, self.hp.dense_activation)
            except AttributeError:
                activation = tf.nn.selu
            self.logits = self.doc_output
            for l in self.hp.dense_layers:
                self.logits = tf.layers.dense(
                    self.logits,
                    l,
                    activation=activation,
                    bias_initializer=tf.contrib.layers.xavier_initializer,
                    kernel_initializer=tf.contrib.layers.xavier_initializer,
                )

            with tf.variable_scope("logits"):
                self.logits = tf.layers.dense(
                    self.logits,
                    self.hp.num_classes,
                    activation=None,
                    bias_initializer=tf.contrib.layers.xavier_initializer,
                    kernel_initializer=tf.contrib.layers.xavier_initializer,
                )

            k_name = [
                n.name
                for n in tf.get_default_graph().as_graph_def().node
                if "logits" in n.name
                and "kernel" in n.name
                and not n.name.split("kernel")[-1]
            ][0]
            kernel = tf.get_default_graph().get_tensor_by_name(k_name + ":0")
            b_name = [
                n.name
                for n in tf.get_default_graph().as_graph_def().node
                if "logits" in n.name
                and "bias" in n.name
                and not n.name.split("bias")[-1]
            ][0]
            bias = tf.get_default_graph().get_tensor_by_name(b_name + ":0")

            tf.summary.histogram("kernel", kernel, collections=["training"])
            tf.summary.histogram("bias", bias, collections=["training"])
            tf.summary.histogram("logits", self.logits, collections=["training"])

            self.prediction = (
                tf.sigmoid(self.logits)
                if self.hp.multilabel
                else tf.argmax(self.logits, axis=1)
            )

    def set_logits(self):
        self.set_lengths()
        self.set_sentence_level()
        self.set_doc_level()
        self.set_classifier()

    def set_embedding_matrix(self, emb_matrix=None, vocab_size=None):
        if vocab_size and not emb_matrix:
            self.hp.set_vocab_size(vocab_size)

        self.np_embedding_matrix = emb_matrix
        if emb_matrix:
            self.embedding_matrix = tf.get_variable(
                "embedding_matrix",
                shape=self.np_embedding_matrix.shape,
                initializer=tf.constant_initializer(self.np_embedding_matrix),
                trainable=self.hp.trainable_embedding_matrix,
            )
        else:
            self.embedding_matrix = tf.get_variable(
                "random_embedding",
                shape=(self.hp.vocab_size, self.hp.embedding_dim),
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True,
            )
