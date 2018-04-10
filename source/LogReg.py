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
        self.input_vector = None

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
            self.input_layer
        )  # batch x doc x sentence x embedding
        self.mean_vector = tf.reduce_mean(
            tf.reduce_mean(
                self.embedded_inputs,
                axis=2
            ),  # batch x doc x embedding
            axis=1
        )  # batch x embedding

        flat_vector = tf.reshape(self.embedded_inputs, shape=(
            self.embedded_inputs.shape[0],
            self.embedded_inputs.shape[1] * self.embedded_inputs.shape[2],
            self.embedded_inputs.shape[3],
        ))  # batch x sent*doc x embedding

        l2_embeddings = tf.norm(flat_vector, axis=2)  # batch x sent*doc
        max_locations = tf.cast(
            tf.argmax(l2_embeddings, axis=1), tf.int32)
        range_tensor = tf.range(tf.shape(max_locations)[0])
        indexes = tf.stack([range_tensor, max_locations], axis=1)
        self.max_vector = tf.gather_nd(
            flat_vector, indexes)  # batch x embedding

        self.input_vector = tf.concat(
            [self.mean_vector, self.max_vector],
            axis=1
        )  # batch x 2*embedding

        self.logits = tf.layers.dense(
            self.input_vector,
            self.hp.num_classes,
            kernel_initializer=tf.initializers.truncated_normal
        )
