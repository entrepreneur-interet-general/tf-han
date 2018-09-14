from .trainer import Trainer
import tensorflow as tf
import numpy as np


class Ensemble:
    def __init__(self):
        self.trainers = []
        self.logits = []

        self.features = []
        self.labels = []

        self.graph = tf.Graph()

    def add_trainer(self, trainer):
        self.trainers.append(trainer)

    def add_trainers(self, trainers):
        for trainer in trainers:
            self.add_trainer(trainer)

    def set_logits(
        self,
        checkpoint_dirs,
        batch_size,
        trainer_types=None,
        hp_names=None,
        hp_exts=None,
        simple_saves=None,
    ):
        trainer_types = trainer_types or ["DST"] * len(checkpoint_dirs)
        hp_names = hp_names or [""] * len(checkpoint_dirs)
        hp_exts = hp_exts or ["json"] * len(checkpoint_dirs)
        simple_saves = simple_saves or ["True"] * len(checkpoint_dirs)
        for i, ckpt in enumerate(checkpoint_dirs):
            self.add_trainer(
                Trainer.restore(
                    ckpt,
                    trainer_types[i],
                    hp_names[i],
                    hp_exts[i],
                    simple_saves[i],
                    graph=self.graph,
                )
            )
            if i == 0:
                self.set_data()

            logits = []
            for b in range(len(self.features) // batch_size):
                logits.append(
                    self.trainers[-1].infer(
                        self.features[
                            len(self.features) * b : len(self.features) * (b + 1)
                        ],
                        with_logits=True,
                    )
                )
            self.logits.append(np.vstack(logits))

    def set_data(self):
        self.features = []
        self.labels = []

        while True:
            try:
                feat, lab = self.trainers[0].sess.run(
                    [self.trainers[0].input_tensor, self.trainers[0].labels_tensor],
                    feed_dict={self.trainers[0].mode_ph: "val"},
                )
                self.features.append(feat)
                self.labels.append(lab)
            except tf.errors.OutOfRangeError:
                break
        self.features = np.vstack(self.features)
        self.labels = np.vstack(self.labels)
