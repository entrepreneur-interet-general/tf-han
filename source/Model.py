import tensorflow as tf


class Model(object):
    def __init__(self, hp, graph=None):
        self.hp = hp
        self.loss = None
        self.logits = None
        self.prediction = None
        self.input_tensor = None
        self.labels_tensor = None

        self.graph = graph or tf.Graph()

    def set_logits(self):
        raise NotImplementedError('set_logits should be implemented')

    def set_embedding_matrix(self):
        raise NotImplementedError('set_embedding_matrix should be implemented')

    def set_loss(self):
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels_tensor,
                name='sig-xent'),
            name="mean-sig-xent")

    def build(self, input_tensor, labels_tensor):
        self.input_tensor = input_tensor
        self.labels_tensor = tf.cast(labels_tensor, tf.float32)
        with self.graph.as_default():
            self.set_embedding_matrix()
            self.set_logits()
            self.set_loss()
