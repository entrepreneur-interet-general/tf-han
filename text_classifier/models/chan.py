import tensorflow as tf
from ..utils.tf_utils import bidirectional_rnn, task_specific_attention
from .han import HAN

MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell


class CHAN(HAN):
    def __init__(self, hp, is_training, graph=None):
        self.hp = hp
        self.word_cell_fw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )
        self.word_cell_bw = MultiRNNCell(
            [self.get_rnn_cell(is_training) for _ in range(self.hp.rnn_layers)]
        )

        self.input_size = None
        self.encoded_words = None
        self.word_lengths = None
        self.words = None

        super().__init__(hp, is_training, graph)

    def set_word_level(self):
        with tf.variable_scope("word-level"):
            with tf.variable_scope("embedding-lookup"):
                self.embedded_inputs = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.input_tensor
                )

            with tf.variable_scope("reshape"):
                self.embedded_words = tf.reshape(
                    self.embedded_inputs,
                    [
                        self.batch * self.docs * self.sents,
                        self.words,
                        self.hp.embedding_dim,
                    ],
                )

                self.input_size = (
                    self.batch
                    * self.docs
                    * self.sents
                    * self.words
                    * self.hp.embedding_dim
                )

                self.word_lengths = tf.reshape(
                    self.word_lengths, [self.batch * self.docs * self.sents]
                )

            with tf.variable_scope("bidir-rnn") as scope:
                self.encoded_words, _ = bidirectional_rnn(
                    self.word_cell_fw,
                    self.word_cell_bw,
                    self.embedded_words,
                    self.word_lengths,
                    scope=scope,
                )

            with tf.variable_scope("attention") as scope:
                self.attended_words = task_specific_attention(
                    self.encoded_words, self.hp.cell_size, scope=scope
                )

            with tf.variable_scope("dropout"):
                self.word_output = tf.contrib.layers.dropout(
                    self.attended_words,
                    keep_prob=self.hp.dropout,
                    is_training=self.is_training,
                )

    def _unstack_lengths(self):
        with tf.variable_scope("unstack-lengths"):
            self.batch, self.docs, self.sents, self.words = tf.unstack(
                tf.shape(self.input_tensor)
            )

    def set_lengths(self):
        self._unstack_lengths()
        with tf.variable_scope("set-lengths"):

            batch_doc_sent_indexes = tf.reduce_sum(self.input_tensor, axis=-1)

            # b x d x s
            self.word_lengths = tf.count_nonzero(self.input_tensor, axis=-1)
            # b x d
            self.sentence_lengths = tf.count_nonzero(batch_doc_sent_indexes, axis=-1)
            # b
            self.doc_lengths = tf.count_nonzero(
                tf.reduce_sum(batch_doc_sent_indexes, axis=-1), axis=-1
            )

    def set_logits(self):
        self.set_lengths()
        self.set_word_level()
        self.set_sentence_level()
        self.set_doc_level()
        self.set_classifier()
