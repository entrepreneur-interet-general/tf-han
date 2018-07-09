import json
import shutil
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants  # pylint: disable=E0611

from ..hyperparameters import HP
from ..models import HAN
from ..utils.tf_utils import streaming_f1
from ..utils.utils import EndOfExperiment


def strtime(ref):
    """Formats a string as the difference between a reference time
    and the current time

    Args:
        ref (float): reference time

    Returns:
        str: h:m:s from time() - ref
    """
    td = int(time() - ref)
    m, s = divmod(td, 60)
    h, m = divmod(m, 60)
    return "{:2}:{:2}:{:2}".format(h, m, s).replace(" ", "0")


class Trainer:
    def __init__(
        self,
        model_type=None,
        hp=None,
        model=None,
        graph=None,
        sess=None,
        restored=False,
    ):
        """Creates a Trainer.

        Args:
            model_type (str): can be a known model or "reuse"
            hp (Hyperparameter): the trainer's params
            model (Model, optional): Defaults to None.
                Model to use if model_type is reuse
            graph (tf.Graph, optional): Defaults to None.
                If None, a new one is created
            sess (tf.Session, optional): Defaults to None.
                If None, a new one is created
            processers ([type], optional): Defaults to None.
                If None, new ones are created

        Raises:
            ValueError: if the model is unknown or if model_type
                is "reuse" but model is None
        """
        self.graph = graph or tf.Graph()
        self.hp = hp or HP()
        self.hp.restored = restored
        self.inferences = []
        self.model_type = model_type or self.hp.model_type
        self.new_procs = True
        self.prepared = False
        self.sess = sess or tf.Session(graph=self.graph)

        self.accuracy = None
        self.features_data_ph = None
        self.global_step_var = None
        self.infer_dataset = None
        self.infer_dataset_init_op = None
        self.infer_iter = None
        self.input_tensor = None
        self.is_infering = None
        self.labels_data_ph = None
        self.labels_tensor = None
        self.learning_rate = None
        self.macro_f1 = None
        self.metrics_updates = None
        self.micro_f1 = None
        self.one_hot_predictions = None
        self.ref_feats = None
        self.ref_labs = None
        self.saver = None
        self.server = None
        self.summary_op = None
        self.train_bs = None
        self.train_data_length = None
        self.train_dataset = None
        self.train_dataset_init_op = None
        self.train_feats = None
        self.train_iter = None
        self.train_labs = None
        self.train_metrics_summary_op = None
        self.train_op = None
        self.train_start_time = None
        self.train_writer = None
        self.training_summary_op = None
        self.val_bs = None
        self.val_dataset = None
        self.val_dataset_init_op = None
        self.val_feats = None
        self.val_iter = None
        self.val_labs = None
        self.val_metrics_summary_op = None
        self.val_size = None
        self.val_writer = None
        self.weighted_f1 = None
        self.words = None
        self.y_pred = None
        self.infered_logits = None

        self.metrics = {}
        self.summary_ops = {}

        if self.model_type == "HAN":
            self.model = HAN(self.hp, self.graph)
        elif self.model_type == "reuse" and model:
            self.model = model
        else:
            raise ValueError("Invalid model")

        if not restored:
            self.set_placeholders()

    def set_placeholders(self):
        with self.graph.as_default():
            self.global_step_var = tf.get_variable(
                "global_step",
                [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32,
                trainable=False,
            )
            self.batch_size_ph = tf.placeholder(tf.int64, name="batch_size_ph")
            self.mode_ph = tf.placeholder(tf.string, name="mode_ph")
            self.model.mode_ph = self.mode_ph
            self.model.is_training = tf.equal(self.mode_ph, "train")

    def save(self, path=None, simple_save=True, ckpt_save=True):
        with self.graph.as_default():
            if simple_save:
                inputs = {
                    "mode_ph": self.mode_ph,
                    "batch_size_ph": self.batch_size_ph,
                    "features_data_ph": self.features_data_ph,
                    "labels_data_ph": self.labels_data_ph,
                }
                outputs = {
                    "prediction": self.model.prediction,
                    "logits": self.model.logits,
                }
                tf.saved_model.simple_save(  # pylint: disable=E1101
                    self.sess,
                    str(self.hp.dir / "checkpoints" / "simple"),
                    inputs,
                    outputs,
                )
                self.hp.dump(to="json")
                ops = {
                    "val_dataset_init_op": self.val_dataset_init_op,
                    "infer_dataset_init_op": self.infer_dataset_init_op,
                    "train_dataset_init_op": self.train_dataset_init_op,
                }
                with open(self.hp.dir / "simple_saved_mapping.json", "w") as f:
                    json.dump(
                        {
                            "inputs": {
                                k: str(v).split('"')[1] for k, v in inputs.items()
                            },
                            "outputs": {
                                k: str(v).split('"')[1] for k, v in outputs.items()
                            },
                            "ops": {k: str(v).split('"')[1] for k, v in ops.items()},
                        },
                        f,
                    )
            if ckpt_save:
                self.saver = tf.train.Saver(tf.global_variables())
                self.saver.save(
                    self.sess,
                    str(self.hp.dir / "checkpoints" / "ckpt"),
                    global_step=self.hp.global_step,
                )

    @staticmethod
    def restore(checkpoint_dir, hp_name="", hp_ext="json", simple_save=True):
        if simple_save:
            checkpoint_dir = Path(checkpoint_dir).resolve()
            hp = HP.load(checkpoint_dir, hp_name, hp_ext)
            trainer = Trainer("HAN", hp, restored=True)
            tf.saved_model.loader.load(
                trainer.sess,
                [tag_constants.SERVING],
                str(checkpoint_dir / "checkpoints" / "simple"),
            )
            mapping_path = checkpoint_dir / "simple_saved_mapping.json"
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            for k, v in mapping["inputs"].items():
                setattr(trainer, k, trainer.graph.get_tensor_by_name(v))
            for k, v in mapping["ops"].items():
                setattr(trainer, k, trainer.graph.get_operation_by_name(v))
            for k, v in mapping["outputs"].items():
                setattr(trainer.model, k, trainer.graph.get_tensor_by_name(v))

        else:
            hp = HP.load(checkpoint_dir, hp_name, hp_ext)
            trainer = Trainer("HAN", hp, restored=True)
            trainer.prepare()
            trainer.saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # pylint: disable=E1101
            trainer.saver.restore(trainer.sess, ckpt.model_checkpoint_path)

        return trainer

    def dump_logits(self):
        if self.infered_logits is not None:
            np.savetxt(
                self.hp.dir / "inferred_logits.csv", self.infered_logits, delimiter=", "
            )
        else:
            raise ValueError("No inferred logits")

    def initialize_uninitialized(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run(
                [tf.is_variable_initialized(var) for var in global_vars]
            )
            not_initialized_vars = [
                v for (v, f) in zip(global_vars, is_not_initialized) if not f
            ]

            if len(not_initialized_vars):
                print("Initializing {} variables".format(len(not_initialized_vars)))
                self.sess.run(tf.variables_initializer(not_initialized_vars))

    def delete(self, ask=True):
        """Delete the trainer's directory after asking for confiirmation

            ask (bool, optional): Defaults to True.
            Whether to ask for confirmation
        """
        if not ask or "y" in input("Are you sure? (y/n)"):
            shutil.rmtree(self.hp.dir)

    def make_iterators(self):
        with self.graph.as_default():
            with tf.variable_scope("datasets"):
                with tf.variable_scope("train"):
                    self.train_iter = tf.data.Iterator.from_structure(
                        self.train_dataset.output_types,
                        self.train_dataset.output_shapes,
                    )
                    self.train_dataset_init_op = self.train_iter.make_initializer(
                        self.train_dataset, name="train_dataset_init_op"
                    )
                with tf.variable_scope("val"):
                    self.val_iter = tf.data.Iterator.from_structure(
                        self.val_dataset.output_types, self.val_dataset.output_shapes
                    )
                    self.val_dataset_init_op = self.val_iter.make_initializer(
                        self.val_dataset, name="val_dataset_init_op"
                    )
                with tf.variable_scope("infer"):
                    self.infer_iter = tf.data.Iterator.from_structure(
                        self.infer_dataset.output_types,
                        self.infer_dataset.output_shapes,
                    )
                    self.infer_dataset_init_op = self.infer_iter.make_initializer(
                        self.infer_dataset, name="infer_dataset_init_op"
                    )
                self.input_tensor, self.labels_tensor = tf.cond(
                    self.model.is_training,
                    self.train_iter.get_next,
                    lambda: tf.cond(
                        tf.equal(self.mode_ph, "infer"),
                        self.infer_iter.get_next,
                        self.val_iter.get_next,
                    ),
                )

    def make_datasets(self):
        raise NotImplementedError("Implement make_datasets")

    def get_input_pair(self, is_val=False):
        self.initialize_iterators(is_val)
        mode = "val" if is_val else "train"
        return self.sess.run(
            [self.input_tensor, self.labels_tensor], feed_dict={self.mode_ph: mode}
        )

    def set_metrics(self):
        with tf.variable_scope("metrics"):
            for scope in ["train", "val"]:
                with tf.variable_scope(scope):
                    with tf.variable_scope("f1"):
                        f1s, f1_updates = streaming_f1(
                            self.labels_tensor,
                            self.model.one_hot_prediction,
                            self.hp.num_classes,
                        )
                        micro_f1, macro_f1, weighted_f1 = f1s
                    with tf.variable_scope("accuracy"):
                        accuracy, accuracy_update = tf.metrics.accuracy(
                            tf.cast(self.model.one_hot_prediction, tf.int32),
                            self.labels_tensor,
                        )
                    self.metrics[scope] = {
                        "accuracy": accuracy,
                        "f1": {
                            "micro": micro_f1,
                            "macro": macro_f1,
                            "weighted": weighted_f1,
                        },
                        "updates": tf.group(f1_updates, accuracy_update),
                    }

    def set_summaries(self):
        with tf.variable_scope("summaries"):
            for scope in self.metrics:
                for metric in self.metrics[scope]:
                    if metric == "accuracy":
                        tf.summary.scalar(
                            "{}".format(metric),
                            self.metrics[scope][metric],
                            collections=[scope + "_metrics"],
                        )
                    elif metric == "f1":
                        for f1 in self.metrics[scope][metric]:
                            tf.summary.scalar(
                                "{}_{}".format(metric, f1),
                                self.metrics[scope][metric][f1],
                                collections=[scope + "_metrics"],
                            )

            tf.summary.scalar("loss", self.model.loss, collections=["training"])
            tf.summary.scalar(
                "learning_rate", self.learning_rate, collections=["training"]
            )
            self.summary_ops["training"] = tf.summary.merge_all(key="training")
            self.summary_ops["val_metrics"] = tf.summary.merge_all(key="val_metrics")
            self.summary_ops["train_metrics"] = tf.summary.merge_all(
                key="train_metrics"
            )

        self.train_writer = tf.summary.FileWriter(
            str(self.hp.dir / "tensorboard" / "train"), self.graph
        )
        self.val_writer = tf.summary.FileWriter(
            str(self.hp.dir / "tensorboard" / "val")
        )

    def reset_metrics(self, mode):
        reset_vars = [
            v for v in tf.local_variables() if "metrics" in v.name and mode in v.name
        ]
        self.sess.run(tf.variables_initializer(reset_vars))

    def build(self):
        """Performs the graph manipulations to create the trainer:
            * builds the model
            * creates the optimizer
            * computes metrics tensors
            * creates summaries for Tensorboard

        Raises:
            ValueError: cannot be run before the datasets are created
        """
        with self.graph.as_default():
            if self.train_dataset is None:
                raise ValueError("Run make_datasets() first!")

            self.model.build(
                self.input_tensor, self.labels_tensor, None, self.hp.vocab_size
            )

            with tf.variable_scope("optimization"):

                if int(self.hp.decay_rate) == 1:
                    self.learning_rate = tf.get_variable(
                        "learning_rate",
                        shape=[],
                        initializer=tf.constant_initializer(self.hp.learning_rate),
                        trainable=False,
                    )
                else:
                    self.learning_rate = tf.train.exponential_decay(
                        self.hp.learning_rate,
                        self.global_step_var,
                        self.hp.decay_steps,
                        self.hp.decay_rate,
                    )
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gradients = tf.gradients(self.model.loss, tf.trainable_variables())
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.hp.max_grad_norm
                )
                grads_and_vars = tuple(zip(clipped_gradients, tf.trainable_variables()))
                self.train_op = optimizer.apply_gradients(
                    grads_and_vars, global_step=self.global_step_var
                )

            self.set_metrics()
            self.set_summaries()

    def eval_tensor(self, tensor_name):
        return self.sess.run(self.graph.get_tensor_by_name(tensor_name))

    def initialize_iterators(self, is_val=False, inference_data=None):
        """Initializes the train and validation iterators from
        the Processers' data

            is_val (bool, optional): Defaults to False. Whether to initialize
                the validation (True) or training (False) iterator
        """
        raise NotImplementedError("Implement initialize_iterators")

    def prepare(self):
        """Runs all steps necessary before training can start:
            * Processers should have their data ready
            * Hyperparameter is updated accordingly
            * Datasets are initialized
            * Trainer is built
        """
        raise NotImplementedError("Implement prepare")

    def infer(self, features, with_logits=True):
        self.initialize_iterators(inference_data=features)
        fd = {self.mode_ph: "infer"}
        prediction = self.model.prediction
        logits = self.model.logits

        if with_logits:
            return logits.eval(session=self.sess, feed_dict=fd)
        return prediction.eval(session=self.sess, feed_dict=fd)

    def validate(self, force=False):
        """Runs a validation step according to the current
        global_step, i.e. if step % hp.val_every_steps == 0

        force (bool, optional): Defaults to False. Whether
            or not to force validation regarless of global_step

        Returns:
            str: Information to print
        """
        # Validate every n examples so every n/batch_size batchs

        if self.hp.global_step == 0:
            # don't validate on first step eventhough
            # self.hp.global_step% val_every_steps == 0
            return None

        if (not force) and self.hp.global_step % self.hp.val_every_steps != 0:
            # validate if force eventhough
            # self.hp.global_step% val_every_steps != 0
            return None

        self.initialize_iterators(is_val=True)
        self.reset_metrics("val")
        self.one_hot_predictions = []
        while True:
            try:
                _, s, acc, mic, mac, wei, pred, logs = self.sess.run(
                    [
                        self.metrics["val"]["updates"],
                        self.summary_ops["val_metrics"],
                        self.metrics["val"]["accuracy"],
                        self.metrics["val"]["f1"]["micro"],
                        self.metrics["val"]["f1"]["macro"],
                        self.metrics["val"]["f1"]["weighted"],
                        self.model.one_hot_prediction,
                        self.model.logits,
                    ],
                    feed_dict={self.mode_ph: "val"},
                )
                self.one_hot_predictions.append(pred)
                self.infered_logits = logs
            except tf.errors.OutOfRangeError:
                break

        self.val_writer.add_summary(s, self.hp.global_step)
        self.val_writer.flush()
        self.train_writer.flush()

        print("\n", np.sum([_.sum(axis=0) for _ in self.one_hot_predictions], axis=0))

        return acc, mic, mac, wei

    def epoch_string(self, epoch, loss, lr, metrics):
        val_string = ""
        if metrics is not None:
            acc, mic, mac, wei = metrics
            val_string = " | Validation metrics: "
            val_string += "Acc {:.4f} Mi {:.4f} Ma {:.4f} We {:.4f}\n".format(
                acc, mic, mac, wei
            )
        epoch_string = "[{}] EPOCH {:2} / {:2} | "
        epoch_string += "Step {:5} | Loss: {:.4f} | "
        epoch_string += "lr: {:.5f} | "
        epoch_string = epoch_string.format(
            strtime(self.train_start_time),
            epoch + 1,
            self.hp.epochs,
            self.hp.global_step,
            loss,
            lr,
        )
        epoch_string += val_string
        return epoch_string

    def train(self, continue_training=False):
        """Main method: the training.
            * Prepares the Trainer
            * For each batch of each epoch:
                * train
                * get loss
                * get global step
                * write summary
                * validate
                * print status
        """
        if self.hp.restored:
            self.hp.retrained = True
        self.train_start_time = time()
        loss, metrics = -1, None
        with self.graph.as_default():
            self.prepare()
            if not continue_training:
                tf.global_variables_initializer().run(session=self.sess)
                tf.tables_initializer().run(session=self.sess)
                self.reset_metrics("train")
            else:
                self.initialize_uninitialized()
            try:
                for epoch in range(self.hp.epochs):
                    self.initialize_iterators(is_val=False)
                    stop = False
                    while not stop:
                        try:
                            loss, lr, summary, _, _ = self.sess.run(
                                [
                                    self.model.loss,
                                    self.learning_rate,
                                    self.summary_ops["training"],
                                    self.train_op,
                                    self.metrics["train"]["updates"],
                                ],
                                feed_dict={self.mode_ph: "train"},
                            )
                            self.hp.global_step = tf.train.global_step(
                                self.sess, self.global_step_var
                            )
                            self.train_writer.add_summary(summary, self.hp.global_step)

                            if self.hp.global_step % 20 == 0:
                                self.train_writer.add_summary(
                                    self.sess.run(
                                        self.summary_ops["train_metrics"],
                                        feed_dict={self.mode_ph: "train"},
                                    ),
                                    self.hp.global_step,
                                )
                                self.train_writer.flush()
                                self.reset_metrics("train")

                            metrics = self.validate()

                        except tf.errors.OutOfRangeError:
                            stop = True
                        finally:
                            print(self.epoch_string(epoch, loss, lr, metrics), end="\r")

                # End of epochs
                if self.hp.global_step % self.hp.val_every_steps != 0:
                    # print final validation if not done at the end
                    # of last epoch
                    metrics = self.validate(force=True)
                    print(
                        "\n[{}] Finally {}".format(
                            strtime(self.train_start_time), metrics
                        )
                    )
                    self.save()
                    print("Saved.")
                return metrics
            except KeyboardInterrupt:
                while True:
                    try:
                        print("\nInterrupting. Save or delete?")
                        answer = input("s/d : ")
                        if "s" in answer:
                            self.save()
                            print("Saved.")
                            break
                        elif "d" in answer:
                            self.delete(False)
                            print("Deleted.")
                        print("\nContinue Experiment?")
                        answer = input("y/n : ")
                        if "y" not in answer:
                            raise EndOfExperiment("Stopping Experiment")
                        else:
                            return metrics
                    except KeyboardInterrupt:
                        raise EndOfExperiment("Stopping Experiment")
