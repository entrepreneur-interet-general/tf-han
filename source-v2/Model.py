import tensorflow as tf


class Model(object):
    """Abstract class for models to feed into a Trainer

    Raises:
        NotImplementedError: set_logits
        NotImplementedError: set_embedding_matrix
    """

    def __init__(self, hp, graph=None):
        """Create a model with attributes:
            * hyperparameter
            * loss
            * prediction
            * input and labels tensors

        Loss is always a sigmoid_cross_entropy_with_logits.
        The child class defines the model's structure through
        set_embedding_matrix and set_logits

        Args:
            hp (Hyperparameter): the model's params
            graph (tf.Graph, optional): Defaults to None.
                If none is provided, a new one is created
        """
        self.hp = hp
        self.loss = None
        self.logits = None
        self.prediction = None
        self.input_tensor = None
        self.labels_tensor = None
        self.one_hot_prediction = None

        self.graph = graph or tf.Graph()

    def set_logits(self):
        """Should define the model's structure
        through its output layer: model.logits which
        will be used to compute loss and predictions

        Raises:
            NotImplementedError: Abstract method
        """
        raise NotImplementedError("set_logits should be implemented")

    def set_embedding_matrix(self, emb_matrix=None, vocab_size=None):
        """Should define how the model will lookup the indexes
        it will get as inputs.

        emb_matrix (np.array, optional): Defaults to None. Embedding matrix
        of shape nb_words x embedding_dim

        Raises:
            NotImplementedError: Abstract method
        """
        raise NotImplementedError("set_embedding_matrix should be implemented")

    def set_loss(self):
        """Creates the loss operation as a component-wise logistic
        regression by applying sigmoid_cross_entropy_with_logits
        """
        with self.graph.as_default():
            with tf.variable_scope("loss"):
                if self.hp.multilabel:
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.logits,
                            labels=self.labels_tensor,
                            name="sig-xent",
                        ),
                        name="mean-sig-xent",
                    )
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.logits, labels=self.labels_tensor, name="xent"
                        ),
                        name="mean-xent",
                    )

    def build(self, input_tensor, labels_tensor, emb_matrix=None, vocab_size=None):
        """Computes the major operations to create a model:
        * set_embedding_matrix
        * set_logits
        * set_loss

        Args:
            input_tensor (tf.Variable): batch x docs x sentences
            labels_tensor (tf.Variable): batch x num_classes
        """
        with self.graph.as_default():
            with tf.variable_scope("model"):
                self.input_tensor = input_tensor
                self.labels_tensor = tf.cast(labels_tensor, tf.float32)
                self.set_embedding_matrix(emb_matrix, vocab_size)
                self.set_logits()
                self.set_loss()
                self.one_hot_prediction = tf.one_hot(
                    self.prediction, self.hp.num_classes
                )
