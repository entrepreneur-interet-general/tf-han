from Model import Model
import tensorflow as tf


class LogReg(Model):

    """
        Baseline Logistic Regression Model
    """

    def __init__(self, hp, graph=None):
        super().__init__(hp, graph)

        self.embedded_inputs = None
        self.mean_vector = None
        self.max_vector = None
        self.logReg_vector = None
        print(repr(self.hp))

    def set_embedding_matrix(self):
        self.embedding_matrix = tf.get_variable(
            'embedding_matrix',
            shape=(self.hp.vocab_size, self.hp.embedding_dim),
            initializer=tf.random_normal_initializer,
            trainable=True
        )

    def set_logits(self):
        self.embedded_inputs = tf.nn.embedding_lookup(
            self.embedding_matrix,
            self.input_tensor
        )  # batch x doc x sentence x embedding

        with tf.name_scope('mean_vector'):
            self.mean_vector = tf.reduce_mean(
                tf.reduce_mean(
                    self.embedded_inputs,
                    axis=2
                ),  # batch x doc x embedding
                axis=1
            )  # batch x embedding

        with tf.name_scope('max_vector'):
            flat_vector = tf.reshape(self.embedded_inputs, shape=(
                tf.shape(self.embedded_inputs)[0], -1,
                tf.shape(self.embedded_inputs)[3]
            ))  # batch x sent*doc x embedding

            l2_embeddings = tf.norm(flat_vector, axis=2)  # batch x sent*doc
            max_locations = tf.cast(
                tf.argmax(l2_embeddings, axis=1), tf.int32)
            range_tensor = tf.range(tf.shape(max_locations)[0])
            indexes = tf.stack([range_tensor, max_locations], axis=1)
            self.max_vector = tf.gather_nd(
                flat_vector, indexes)  # batch x embedding

        with tf.name_scope('logReg_vector'):
            self.logReg_vector = tf.reshape(
                tf.concat(
                    [self.mean_vector, self.max_vector],
                    axis=1
                ), shape=(-1, 2*self.hp.embedding_dim)
            )  # batch x 2*embedding
        with tf.name_scope('dense_layer'):
            self.logits = tf.layers.dense(
                self.logReg_vector,
                self.hp.num_classes,
                kernel_initializer=tf.initializers.truncated_normal
            )

        self.prediction = tf.nn.sigmoid(self.logits, name='prediction')
