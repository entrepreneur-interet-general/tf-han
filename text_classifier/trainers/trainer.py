import json
import shutil
import threading
from pathlib import Path
from time import time
import collections

import numpy as np
import tensorflow as tf
from scipy.special import expit
from tensorflow.python.saved_model import tag_constants  # pylint: disable=E0611

from ..hyperparameters import HP
from ..models import HAN
from ..models import CHAN
from ..utils.tf_utils import streaming_f1
from ..utils.utils import EndOfExperiment, thread_eval
from .base_trainer import BaseTrainer


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


class Trainer(BaseTrainer):
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
        self.infered_logits = None  # logits inferred at validation time
        self.infered_labels = None  # target labels for these logits
        self.stop_learning_dic = collections.defaultdict(int)

        self.metrics = {}
        self.summary_ops = {}

        if not restored:
            self.set_placeholders()

        if self.model_type == "HAN":
            self.model = HAN(self.hp, self.is_training, self.graph)
        elif self.model_type == "CHAN":
            self.model = CHAN(self.hp, self.is_training, self.graph)
        elif self.model_type == "reuse" and model:
            self.model = model
        else:
            raise ValueError("Invalid model")

    def set_placeholders(self):
        """Creates the trainer's placeholders:
        the global_step, batch_size, mode_ph and creates the is_training
        tensor.
        """
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
            self.is_training = tf.equal(self.mode_ph, "train")

    def save(self, simple_save=True, ckpt_save=True):
        """Saves the trainer's graph's session's values in 
            
            trainer.hp.dir / checkpoints / simple

            trainer.hp.dir / checkpoints / ckpt

            simple_save (bool, optional): Defaults to True. Use the simple_saver or not
            ckpt_save (bool, optional): Defaults to True. Use the regular ckpt saver 
                or not
        """
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
                    str(
                        self.hp.dir
                        / "checkpoints"
                        / "simple"
                        / str(self.hp.global_step)
                    ),
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
    def restore(
        checkpoint_dir,
        trainer_type=None,
        hp_name="",
        hp_ext="json",
        simple_save=True,
        graph=None,
    ):
        """Restore a previously saved Trainer

        Args:
            checkpoint_dir (str or pathlib.Path): where to find the files ; same as
            trainer_type ( str, optional): Defaults to None. If None, then regular
                trainer is used. Otherwise one of the existing classes of Trainers:
                DST, FT_DST or CDST
            hp_name (str, optional): Defaults to "". The hyperparameters name, as
                `name_hp.json`
            hp_ext (str, optional): Defaults to "json". How the hp was saved
            simple_save (bool, optional): Defaults to True. Which restoring method to 
                use, according to the saving method
            graph (tf.Graph, optional): Defaults to None. Whether to restore the values 
                in an existing graph

        Returns:
            Trainer: the prepared and restored trainer
        """
        from .dataset_trainer import DST
        from .fast_text_dataset_trainer import FT_DST
        from .char_dataset_trainer import CDST

        checkpoint_dir = Path(checkpoint_dir).resolve()
        hp = HP.load(checkpoint_dir, hp_name, hp_ext)
        if trainer_type == "DST":
            trainer = DST("HAN", hp, restored=True, graph=graph)
        elif trainer_type == "FT_DST":
            trainer = FT_DST("HAN", hp, restored=True, graph=graph)
        elif trainer_type == "CDST":
            trainer = CDST("HAN", hp, restored=True, graph=graph)
        else:
            trainer = Trainer("HAN", hp, restored=True, graph=graph)

        trainer.prepare()

        if simple_save:
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
            trainer.saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # pylint: disable=E1101
            trainer.saver.restore(trainer.sess, ckpt.model_checkpoint_path)

        return trainer

    def dump_logits(self, step=None, _eval=True):
        """Saves numpy arrays on disk if there are inferred logits (from validate())
        Otherwise creates an empty file stating no logits were found. From these,
        an exploration of F1 scores at different thresholds may be run in threads
        depending on _eval

            step (int, optional): Defaults to None. Current training step
            eval (bool, optional): Defaults to True. Whether or not to proceed
                to F1-score measurements at differrent thresholds
        """

        step_string = "_{}".format(step or "final")

        if self.infered_logits is not None:
            np.save(
                self.hp.dir / "inferred_logits{}.npy".format(step_string),
                self.infered_logits,
            )
            np.save(
                self.hp.dir / "infered_labels{}.npy".format(step_string),
                self.infered_labels,
            )
        else:
            open(self.hp.dir / "no_inferred_logits.txt", "a").close()
        if _eval:
            preds = expit(self.infered_logits)
            ys = self.infered_labels
        try:
            thread = threading.Thread(
                target=thread_eval,
                args=(
                    ys,
                    preds,
                    self.hp.dir,
                    step,
                    self.hp.ref_labels,
                    self.hp.ref_labels_delimiter,
                ),
            )
            thread.start()
        except Exception as e:
            print(e)

    def initialize_uninitialized(self):
        """Helper function to initialize variables which have not yet
        been initialized.
        """
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run(
                [tf.is_variable_initialized(var) for var in global_vars]
            )
            not_initialized_vars = [
                v for (v, f) in zip(global_vars, is_not_initialized) if not f
            ]

            if not_initialized_vars:
                print("Initializing {} variables".format(len(not_initialized_vars)))
                self.sess.run(tf.variables_initializer(not_initialized_vars))

    def delete(self, ask=True):
        """Delete the trainer's directory after asking for confiirmation

            ask (bool, optional): Defaults to True.
            Whether to ask for confirmation or force deletion
        """
        if not ask or "y" in input("Are you sure? (y/n)"):
            shutil.rmtree(self.hp.dir, ignore_errors=True)

    def make_iterators(self):
        """Creates the train/val/infer iterators from their
        respective datasets and creates the related init ops
        """
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

    def set_input_tensors(self):
        """Creates input_tensor and labels_tensor as a condition on whether
        the Trainer is in training/validation or inference mode:
            depending on mode_ph the tensors will come from the relevant
            iterator.get_next()
        """
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
        """This operation should create the train/val/infer datasets
        from which the iterators will be created in turn.
        
        Raises:
            NotImplementedError: Implement make_datasets
        """
        raise NotImplementedError("Implement make_datasets")

    def get_input_pair(self, is_val=False, batch_size=None):
        """Sample `batch_size` examples from the train or val dataset

            is_val (bool, optional): Defaults to False. Whether to sample from

            batch_size (int, optional): Defaults to None. If no value is specified,
                trainer.hp.batch_size will be used.
        
        Returns:
            list: Length 2 list with the input and target values
        """
        self.initialize_iterators(is_val, batch_size=batch_size)
        mode = "val" if is_val else "train"
        return self.sess.run(
            [self.input_tensor, self.labels_tensor], feed_dict={self.mode_ph: mode}
        )

    def set_metrics(self):
        """Creates a dictionnary storing the train and val metrics + their
        update ops.
        Metrics are accuracy and f1 scores (micro macro weighted)

        trainer.metrics = {
            "train" : {
                "accuracy" : tensor,
                "f1": {
                    "micro": tensor,
                    "macro": tensor,
                    "weighted": tensor
                },
                "updates": operation
            },
            "val": idem
        }

        """
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
        """Create the scalar summaries for the train and val metrics.
        Also, adds summaries for the loss, global_grad_norm before clipping,
        and learning rate (as it may be decayed). (> `training` summary collection)

        Finally, it creates the train and val FileWriter in
            trainer.hp.dir / tensorboard
        """
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
                        for f1_type in self.metrics[scope][metric]:
                            tf.summary.scalar(
                                "{}_{}".format(metric, f1_type),
                                self.metrics[scope][metric][f1_type],
                                collections=[scope + "_metrics"],
                            )

            tf.summary.scalar("loss", self.model.loss, collections=["training"])
            tf.summary.scalar(
                "learning_rate", self.learning_rate, collections=["training"]
            )
            tf.summary.scalar(
                "global_grad_norm", self.global_norm, collections=["training"]
            )
            for g, v in self.grads_and_vars:
                tf.summary.histogram(v.name, v, collections=["training"])
                tf.summary.histogram(v.name + "_grad", g, collections=["training"])

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
        """Resets (to 0) the train or val metrics
        
        Args:
            mode (str): train or val mertrics to reset
        """

        assert mode in {"train", "val"}
        reset_vars = [
            v for v in tf.local_variables() if "metrics" in v.name and mode in v.name
        ]
        self.sess.run(tf.variables_initializer(reset_vars))

    def build(self):
        """Performs the graph manipulations to create the trainer:
            * builds the model
            * creates the optimizer
            * clips the gradient
            * computes the metrics' tensors
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

                gradients, variables = zip(
                    *optimizer.compute_gradients(self.model.loss)
                )

                self.grads_and_vars = zip(gradients, variables)

                clipped_gradients, self.global_norm = tf.clip_by_global_norm(
                    gradients, self.hp.max_grad_norm
                )

                self.train_op = optimizer.apply_gradients(
                    zip(clipped_gradients, variables), global_step=self.global_step_var
                )

            self.set_metrics()
            self.set_summaries()

    def eval_tensor(self, tensor_name):
        """Helper function to infer a tensor's value from its name in the graph
        
        Args:
            tensor_name (str): the tensor's name
        
        Returns:
            np.ndarray: the session's value for this tensor
        """
        return self.sess.run(self.graph.get_tensor_by_name(tensor_name))

    def initialize_iterators(self, is_val=False, inference_data=None, batch_size=None):
        """Initializes the train and validation iterators from
        the Processers' data

            is_val (bool, optional): Defaults to False. Whether to initialize
                the validation (True) or training (False) iterator
        """
        raise NotImplementedError("Implement initialize_iterators")

    def prepare(self):
        """Runs all steps necessary before training can start:
            * Setting up the input flow (make datasets and iterators for instance)
            * Building the model's graph
            * Adding optimization and metrics

        After prepare() all variables in the trainer's graph should be ready to
        be initialized.
        """
        raise NotImplementedError("Implement prepare")

    def infer(self, features, with_logits=True):
        """Get the prediction or logits from input features for the current session.
        
        Args:
            features (np.ndarray(np.int64)): input values to pass forward through the graph,
                i.e. word indices in a vocabulary
            with_logits (bool, optional): Defaults to True. Whether to infer logits or
                the thresholded values
        
        Returns:
            np.ndarray: the output layer's values
        """
        self.initialize_iterators(inference_data=features)
        fd = {self.mode_ph: "infer"}
        if with_logits:
            return self.model.logits.eval(session=self.sess, feed_dict=fd)
        return self.model.prediction.eval(session=self.sess, feed_dict=fd)

    def training_should_stop(self, epoch, val):
        """Returns whether or not the training should stop based on 
        current validation metrics compared to stop_metrics:

        hp.stop_metrics = {
            metric: {
                epoch: when to start watching for the metric
                value: minimum authorized value to continue training, -1 for NaN
                max: maximum occurances of `value` from epoch `epoch`
            }
        }

        Training is stopped if one of the conditions is met.

        hp.stop_metrics = {
            "accuracy": {
                "epoch": 3,
                "value": 0.95,
                "max": 2
            }
        } -> means "from epoch 3 onwards, the training will be stopped the second time
        accuracy is below 95%"
        
        Args:
            epoch (int): current epoch
            val (tuple): current validation metrics: non_zeros, acc, mic, mac, wei
        
        Returns:
            bool: stop the training or not
        """
        non_zeros, *metrics = val

        if not self.hp.stop_learning:
            return False

        if self.hp.stop_metrics:
            assert isinstance(self.hp.stop_metrics, dict)
            acc, mic, mac, wei = metrics
            ref = {
                "accuracy": acc,
                "micro": mic,
                "macro": mac,
                "weighted": wei,
                "non_zero": non_zeros,
            }
            for met, dic in self.hp.stop_metrics.items():

                if "epoch" not in dic:
                    max_epoch = 1
                else:
                    max_epoch = dic["epoch"]

                if epoch + 1 >= max_epoch:
                    stop_value = dic["value"]

                    if stop_value == -1:
                        if np.isnan(ref[met]):
                            self.stop_learning_dic[met] += 1
                    elif ref[met] < stop_value:
                        self.stop_learning_dic[met] += 1

                if "max" in dic:
                    if self.stop_learning_dic[met] == dic["max"]:
                        return True
                elif self.stop_learning_dic[met] == 1:
                    return True

        return False

    def validate(self, force=False):
        """Runs a validation step according to the current
        global_step, i.e. if step % hp.val_every_steps == 0

        For clarity of the training loop, this function performs the check
        of whether or not validation should be run.

        If validation is performed, 5 metrics are computed:
             3 multilabel f1 scores
             accuracy
             non_zeros -> number of classes with a non-zero positive count
                for this validation round.

        So for threshold 0.5 and 5 classes [0 0 0 0 0 0 12 0 1 0] would mean
        that for the whole validation set, 13 samples have a positive count but spanning
        only 2 classes. non_zeros = 2

        force (bool, optional): Defaults to False. Whether
            or not to force validation regarless of global_step

        Returns:
            Tuple or None: None <-> no validation performed
                otherwise tupe is non_zeros, acc, mic, mac, wei
        """

        # Validate every n batchs
        if (not force) and self.hp.global_step == 0:
            # don't validate on first step eventhough
            # self.hp.global_step% val_every_steps == 0
            return None

        if (not force) and self.hp.val_every_steps <= 0:
            # val_every_steps = -1 -> val at epoch ends only
            return None

        if (not force) and self.hp.global_step % self.hp.val_every_steps != 0:
            # validate if force eventhough
            # self.hp.global_step% val_every_steps != 0
            return None

        self.initialize_iterators(is_val=True, batch_size=self.hp.val_batch_size)
        self.reset_metrics("val")
        self.one_hot_predictions = []
        self.infered_logits = []
        self.infered_labels = []
        while True:
            try:
                _, s, acc, mic, mac, wei, pred, logs, ys = self.sess.run(
                    [
                        self.metrics["val"]["updates"],
                        self.summary_ops["val_metrics"],
                        self.metrics["val"]["accuracy"],
                        self.metrics["val"]["f1"]["micro"],
                        self.metrics["val"]["f1"]["macro"],
                        self.metrics["val"]["f1"]["weighted"],
                        self.model.one_hot_prediction,
                        self.model.logits,
                        self.labels_tensor,
                    ],
                    feed_dict={self.mode_ph: "val"},
                )
                self.one_hot_predictions.append(pred)
                self.infered_logits.append(logs)
                self.infered_labels.append(ys)
            except tf.errors.OutOfRangeError:
                break
        self.infered_logits = np.vstack(self.infered_logits)
        self.infered_labels = np.vstack(self.infered_labels)
        self.val_writer.add_summary(s, self.hp.global_step)
        self.val_writer.flush()
        self.train_writer.flush()
        preds = np.sum([_.sum(axis=0) for _ in self.one_hot_predictions], axis=0)
        print("\n", preds)

        non_zeros = np.count_nonzero(preds)

        return non_zeros, acc, mic, mac, wei

    def step_string(self, epoch, loss, lr, metrics):
        """Returns a string to be printed and/or logged expressing
        the current state of the training:
        time elapsed, eppoch and step numbers, loss and learning rate
        + validation metrics at the end of a validation process
        
        Args:
            epoch (int): epoch number
            loss (float): current loss
            lr (float): current learning rate
            metrics (tuple): acc, mic, mac, wei
        
        Returns:
            str : the step string
        """
        val_string = ""
        if metrics is not None:
            acc, mic, mac, wei = metrics
            val_string = " | Validation metrics: "
            val_string += "Acc {:.4f} Mi {:.4f} Ma {:.4f} We {:.4f}".format(
                acc, mic, mac, wei
            )
        step_string = "[{}] EPOCH {:2} / {:2} | "
        step_string += "Step {:5} | Loss: {:.4f} | "
        step_string += "lr: {:.5f} | "
        step_string = step_string.format(
            strtime(self.train_start_time),
            epoch + 1,
            self.hp.epochs,
            self.hp.global_step,
            loss,
            lr,
        )
        step_string += val_string
        return step_string

    def train(self, continue_training=False):
        """Main method: the training.

            * Prepares the Trainer
            * For each batch of each epoch:

                - train
                - get loss
                - get global step
                - write summary
                - validate (returns None if not the right moment according to hp)
                - print status

        continue_training (bool, optional): Defaults to False. 
            Whether or not to initialize all variables or only the unititialized

        Raises:
            EndOfExperiment: Signal that the Experiment should stop

        Returns:
            tuple(float): metrics from the last validation
        """

        if self.hp.restored:
            self.hp.retrained = True

        self.train_start_time = time()

        loss, metrics = -1, None

        with self.graph.as_default():
            # Catch keyboard interrupts
            try:
                # Set up the Graph (input pipeline + model)
                self.prepare()

                # initialize the variables
                if not continue_training:
                    tf.global_variables_initializer().run(session=self.sess)
                    tf.tables_initializer().run(session=self.sess)
                    self.reset_metrics("train")
                else:
                    self.initialize_uninitialized()

                # Main training loop
                for epoch in range(self.hp.epochs):
                    # Feed the iterators with training data
                    self.initialize_iterators(is_val=False)
                    stop = False
                    # While there is data in the training dataset
                    while not stop:
                        try:

                            force_validation = False

                            # 1 training step
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
                            # Update hp's global_step
                            self.hp.global_step = tf.train.global_step(
                                self.sess, self.global_step_var
                            )
                            # Add session's training values to summary
                            # i.e. loss, gradient norm, lr, gradient histograms
                            self.train_writer.add_summary(summary, self.hp.global_step)

                            # Every 20 steps compute training metrics (acc mic mac wei)
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

                        except tf.errors.OutOfRangeError:
                            # End of epoch
                            stop = True
                            force_validation = self.hp.val_every_steps <= 0

                        # Every step run validate() which will return None if it's not
                        # time for validation.
                        val = self.validate(force_validation)

                        if val is not None:  # validation was run
                            # save inferred logits and target labels
                            # for potential further investigation after training
                            # without reloading the whole trainer
                            self.dump_logits(self.hp.global_step)
                            # Save the current graph's session
                            self.save()
                            # According to current metrics training may have to stop
                            # because it is assumed that it is not on a productive path
                            if self.training_should_stop(epoch, val):
                                return metrics
                        else:
                            metrics = None

                        print(self.step_string(epoch, loss, lr, metrics), end="\r")

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

                try:
                    # Trainign loop was interrupted by user.
                    # Save, delete, or do nothing (s/d/anything else)
                    print("\nInterrupting. Save or delete?")
                    answer = input("S / D : ")
                    if "s" in answer.lower():
                        self.save()
                        print("Saved.")
                    elif "d" in answer.lower():
                        self.delete(False)
                        print("Deleted.")

                    # Was the interruption for the trainer or the
                    # experiment?
                    # y -> continue with new trainer
                    # anything else -> stop experiment
                    print("\nContinue Experiment?")
                    answer = input("y/n : ")
                    if "y" not in answer:
                        raise EndOfExperiment("Stopping Experiment")
                    else:
                        return metrics
                except KeyboardInterrupt:
                    raise EndOfExperiment("Stopping Experiment")
