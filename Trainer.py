import tensorflow as tf


class Trainer(object):
    def __init__(self, model, trainer_hp):
        self.model = model
        self.hp = trainer_hp
        self.graph = tf.Graph()

    def train(self):
        with self.graph.as_default():

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.hp.checkpoint_dir,
                    scaffold=None,
                    hooks=None
            ) as sess:
