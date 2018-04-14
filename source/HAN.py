import tensorflow as tf
import numpy as np
from Model import Model
from utils import bidirectional_rnn, task_specific_attention


class HAN(Model):

    def __init__(self, hp, graph=None):
        super().__init__(hp, graph)
        self.embedding_matrix = None
        self.batch = None
        self.docs = None
        self.sents = None
        self.cell = tf.nn.rnn_cell.GRUCell(self.hp.cell_size)
        self.val_ph = None

    def set_logits(self):
        (self.batch,
         self.docs,
         self.sents) = tf.unstack(tf.shape(self.input_tensor))

        self.embedded_inputs = tf.nn.embedding_lookup(
            self.embedding_matrix,
            self.input_tensor
        )

        self.embedded_sentences = tf.reshape(
            self.embedded_inputs,
            [self.batch * self.docs, self.sents, self.hp.embedding_dim]
        )

        self.sentence_lengths = tf.reshape(
            self.sentence_lengths,
            [self.batch * self.docs]
        )

        self.encoded_sentences, _ = bidirectional_rnn(
            self.cell, self.cell, self.embedded_sentences,
            self.sentence_lengths
        )

        self.attended_sentences = task_specific_attention(
            self.encoded_sentences,
            self.hp.cell_size)

        self.sentence_output = tf.contrib.layers.dropout(
            self.attended_sentences, keep_prob=self.keep_prob_dropout_ph,
            is_training=not self.val_ph,
        )

    def get_embedding_matrix(self):
        return np.random.randn(10000, 300)

    def set_embedding_matrix(self):
        """Creates a random trainable embedding matrix
        """
        self.embedding_matrix = tf.get_variable(
            'embedding_matrix',
            shape=(self.hp.vocab_size, self.hp.embedding_dim),
            initializer=tf.constant_initializer(
                self.get_embedding_matrix()
            ),
            trainable=self.hp.trainable_embedding_matrix
        )
