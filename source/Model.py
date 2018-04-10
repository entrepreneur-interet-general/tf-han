import tensorflow as tf


class Model(object):
    def __init__(self, hp):
        self.hp = hp
        self.loss = None
        self.logits = None

        self.output_labels = tf.placeholder(
            tf.float32,
            shape=(
                self.hp.batch_size,
                self.hp.num_classes
            ),
            name="output_labels"
        )
        self.input_layer = tf.placeholder(
            tf.float32,
            shape=(
                self.hp.batch_size,
                self.hp.max_doc_len
                self.hp.max_sent_len,
            )
        )

    def set_logits(self):
        raise NotImplementedError('set_logits should be implemented')

    def set_embedding_matrix(self):
        raise NotImplementedError('set_embedding_matrix should be implemented')

    def set_loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.output_labels,
                name='sig-xent'),
            name="mean-sig-xent")

    def build(self):
        self.set_embedding_matrix()
        self.set_logits()
        self.set_loss()
