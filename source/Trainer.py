from HAN import HAN
from LogReg import LogReg
from Processer import Processer
import tensorflow as tf
from pathlib import Path
from time import time
import shutil
import numpy as np
import Hyperparameter as hyp
import json
from tensorflow.python.saved_model import tag_constants
from utils import f1_score
from importlib import reload

reload(hyp)


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
    return "{:2}:{:2}:{:2}".format(
        h, m, s
    ).replace(' ', '0')


class Trainer(object):

    @staticmethod
    def initialize_from_trainer(trainer):
        """Creates a new Trainer instance from an
        existing trainer's hp and processers

        Args:
            trainer (Trainer): trainer to initialize from

        Returns:
            Trainer: New trainer with identical parameters and
            processers
        """
        hp = trainer.hp
        hp.assign_new_dir()
        return Trainer(
            hp, trainer.model_type, processers=[
                trainer.train_proc, trainer.val_proc])

    def __init__(self,
                 model_type,
                 trainer_hp=None,
                 model=None,
                 graph=None,
                 sess=None,
                 processers=None,
                 restored=False):
        """Creates a Trainer.

        Args:
            model_type (str): can be a known model or "reuse"
            trainer_hp (Hyperparameter): the trainer's params
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
        self.model_type = model_type
        self.hp = trainer_hp or hyp.HP()
        self.graph = graph or tf.Graph()
        self.sess = sess or tf.Session(graph=self.graph)
        self.prepared = False
        self.inferences = []
        self.hp.restored = restored
        self.new_procs = True

        self.server = None
        self.summary_op = None
        self.train_op = None
        self.features_data_ph = None
        self.labels_data_ph = None
        self.val_iter = None
        self.val_dataset = None
        self.train_iter = None
        self.train_dataset = None
        self.infer_iter = None
        self.infer_dataset = None
        self.input_tensor = None
        self.labels_tensor = None
        self.learning_rate = None
        self.y_pred = None
        self.train_writer = None
        self.val_writer = None
        self.global_step_var = None
        self.train_start_time = None
        self.accuracy = None
        self.micro_f1 = None
        self.macro_f1 = None
        self.weighted_f1 = None
        self.val_feats = None
        self.val_labs = None
        self.val_bs = None
        self.train_feats = None
        self.train_labs = None
        self.train_bs = None
        self.saver = None
        self.ref_feats = None
        self.ref_labs = None
        self.val_dataset_init_op = None
        self.infer_dataset_init_op = None
        self.train_dataset_init_op = None

        if model_type == 'LogReg':
            self.model = LogReg(self.hp, self.graph)
        elif model_type == 'HAN':
            self.model = HAN(self.hp, self.graph)
        elif model_type == 'reuse' and model:
            self.model = model
        else:
            raise ValueError('Invalid model')

        if processers:
            assert isinstance(processers, list)
            assert len(processers) == 2

            if isinstance(processers[0], (str, Path)):
                processers = [Path(p) for p in processers]
                self.train_proc = Processer.load(processers[0])
                self.val_proc = Processer.load(processers[1])
            else:
                print('Reused procs')
                self.train_proc, self.val_proc = processers

            self.new_procs = False

        else:
            self.train_proc = Processer()
            self.val_proc = Processer()
        if not restored:
            self.set_placeholders()

    def set_placeholders(self):
        with self.graph.as_default():
            self.global_step_var = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32,
                trainable=False
            )
            self.batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
            self.mode_ph = tf.placeholder(tf.int32, name='mode_ph')
            self.shuffle_val_ph = tf.placeholder(
                tf.bool, name='shuffle_val_ph')
            self.model.mode_ph = self.mode_ph
            self.model.is_training = tf.equal(self.mode_ph, 0)

    def save(self, path=None, simple_save=True, ckpt_save=True):
        with self.graph.as_default():
            if simple_save:
                inputs = {
                    "mode_ph": self.mode_ph,
                    "batch_size_ph": self.batch_size_ph,
                    "features_data_ph": self.features_data_ph,
                    "labels_data_ph": self.labels_data_ph,
                    "shuffle_val_ph": self.shuffle_val_ph,
                }
                outputs = {
                    "prediction": self.model.prediction,
                    "logits": self.model.logits
                }
                tf.saved_model.simple_save(
                    self.sess, str(self.hp.dir / 'checkpoints' /
                                   'simple'), inputs, outputs
                )
                self.hp.dump(to='json')
                ops = {
                    'val_dataset_init_op': self.val_dataset_init_op,
                    'infer_dataset_init_op': self.infer_dataset_init_op,
                    'train_dataset_init_op': self.train_dataset_init_op
                }
                with open(self.hp.dir / 'simple_saved_mapping.json', "w") as f:
                    json.dump({
                        'inputs': {
                            k: str(v).split('"')[1] for k, v in inputs.items()
                        },
                        'outputs': {
                            k: str(v).split('"')[1] for k, v in outputs.items()
                        },
                        'ops': {
                            k: str(v).split('"')[1] for k, v in ops.items()
                        }}, f)
            if ckpt_save:
                self.saver = tf.train.Saver(tf.global_variables())
                self.saver.save(
                    self.sess,
                    str(self.hp.dir / 'checkpoints' / 'ckpt'),
                    global_step=self.hp.global_step)

    @staticmethod
    def restore(checkpoint_dir, hp_name='', hp_ext='json', simple_save=True):
        if simple_save:
            checkpoint_dir = Path(checkpoint_dir)
            hp = hyp.HP.load(checkpoint_dir, hp_name, hp_ext)
            trainer = Trainer('HAN', hp, restored=True)
            tf.saved_model.loader.load(
                trainer.sess,
                [tag_constants.SERVING],
                str(checkpoint_dir / 'checkpoints' / 'simple')
            )
            mapping_path = checkpoint_dir / 'simple_saved_mapping.json'
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            for k, v in mapping['inputs'].items():
                setattr(trainer, k, trainer.graph.get_tensor_by_name(v))
            for k, v in mapping['ops'].items():
                setattr(trainer, k, trainer.graph.get_operation_by_name(v))
            for k, v in mapping['outputs'].items():
                setattr(trainer.model, k, trainer.graph.get_tensor_by_name(v))

        else:
            hp = hyp.HP.load(checkpoint_dir, hp_name, hp_ext)
            trainer = Trainer('HAN', hp, restored=True)
            trainer.prepare()
            trainer.saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            trainer.saver.restore(trainer.sess, ckpt.model_checkpoint_path)

        return trainer

    def initialize_uninitialized(self):
        with self.graph.as_default():
            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(
                global_vars, is_not_initialized) if not f]

            if len(not_initialized_vars):
                print('Initializing {} variables'.format(
                    len(not_initialized_vars)))
                self.sess.run(tf.variables_initializer(not_initialized_vars))

    def delete(self, ask=True):
        """Delete the trainer's directory after asking for confiirmation

            ask (bool, optional): Defaults to True.
            Whether to ask for confirmation
        """
        if not ask or 'y' in input('Are you sure? (y/n)'):
            shutil.rmtree(self.hp.dir)

    def translate(self, batch):
        docs = []
        for d in batch:
            sentences = []
            for s in d:
                if s.sum() > 0:
                    sentences.append(
                        [self.train_proc.reversed_vocab[w]
                         for w in s if w > 0]
                    )
            docs.append(sentences)
        return docs

    def display(self, batch, labels=None):
        if labels is None:
            labels = [None] * len(batch)
        elif len(labels.shape) > 1:
            labels = np.where(labels > 0)[1]
        docs = self.translate(batch)
        return '###\n\n'.join(
            ['Rating: {}'.format(labels[i]) + '\n'.join(
                [' '.join([w for w in s]) for s in d]
            ) for i, d in enumerate(docs)]
        )

    def set_procs(self):
        voc = None
        if self.hp.embedding_file:
            voc, mat = self.train_proc.build_word_vector_matrix(
                self.hp.embedding_file, self.hp.max_words
            )
            self.train_proc.np_embedding_matrix = mat
            voc = {
                k: v + 2 for k, v in voc.items()
            }
            voc['<PAD>'] = 0
            voc['<OOV>'] = 1
            self.train_proc.vocab = voc

        self.train_proc.process(
            data_path=self.hp.train_data_path,
            vocabulary=voc
        )
        self.val_proc.process(
            data_path=self.hp.val_data_path,
            vocabulary=self.train_proc.vocab,
            max_sent_len=self.train_proc.max_sent_len,
            max_doc_len=self.train_proc.max_doc_len
        )

    def save_procs(self, path=None):
        path = path or self.hp.dir
        path = Path(path)
        self.train_proc.save(path / 'train_proc.pkl')
        self.val_proc.save(path / 'val_proc.pkl')

    def make_datasets(self):
        """Creates 2 datasets, one to train the other to validate the model.
        Conditioned on the value of mode_ph, the input and label tensors come
        from either the training set (0), validation set (1) or custom
        inference values (2).
        """

        with self.graph.as_default():
            self.features_data_ph = tf.placeholder(
                tf.int32,
                [None, None, self.train_proc.max_sent_len],
                'features_data_ph'
            )
            self.labels_data_ph = tf.placeholder(
                tf.int32,
                [None, self.hp.num_classes],
                'labels_data_ph'
            )
            with tf.variable_scope('datasets'):
                with tf.variable_scope('train'):
                    self.train_dataset = tf.data.Dataset.from_tensor_slices(
                        (self.features_data_ph, self.labels_data_ph)
                    )
                    self.train_dataset = self.train_dataset.shuffle(
                        buffer_size=100000)
                    self.train_dataset = self.train_dataset.batch(
                        self.batch_size_ph)
                    self.train_iter = tf.data.Iterator.from_structure(
                        self.train_dataset.output_types,
                        self.train_dataset.output_shapes
                    )
                    self.train_dataset_init_op = \
                        self.train_iter.make_initializer(
                            self.train_dataset,
                            name='train_dataset_init_op'
                        )

                with tf.variable_scope('val'):
                    self.val_dataset = tf.data.Dataset.from_tensor_slices(
                        (self.features_data_ph, self.labels_data_ph)
                    )
                    self.val_dataset = self.val_dataset.shuffle(
                            buffer_size=100000)
                    self.val_dataset = self.val_dataset.batch(
                        self.batch_size_ph)
                    self.val_iter = tf.data.Iterator.from_structure(
                        self.val_dataset.output_types,
                        self.val_dataset.output_shapes
                    )
                    self.val_dataset_init_op = \
                        self.val_iter.make_initializer(
                            self.val_dataset,
                            name='val_dataset_init_op'
                        )
                with tf.variable_scope('infer'):
                    self.infer_dataset = tf.data.Dataset.from_tensor_slices(
                        (self.features_data_ph, self.labels_data_ph)
                    )
                    self.infer_dataset = self.infer_dataset.batch(
                        self.batch_size_ph)
                    self.infer_iter = tf.data.Iterator.from_structure(
                        self.infer_dataset.output_types,
                        self.infer_dataset.output_shapes
                    )
                    self.infer_dataset_init_op = \
                        self.infer_iter.make_initializer(
                            self.infer_dataset,
                            name='infer_dataset_init_op'
                        )

                self.input_tensor, self.labels_tensor = tf.cond(
                    tf.equal(self.mode_ph, 0),
                    self.train_iter.get_next,
                    lambda: tf.cond(
                        self.shuffle_val_ph,
                        self.val_iter.get_next,
                        self.infer_iter.get_next
                    )
                )

    def get_input_pair(self, is_val=False, shuffle_val=True):
        self.initialize_iterators(is_val)
        return self.sess.run(
            [self.input_tensor, self.labels_tensor],
            feed_dict={
                self.mode_ph: int(is_val),
                self.shuffle_val_ph: True
            }
        )

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
                raise ValueError('Run make_datasets() first!')

            self.model.build(self.input_tensor, self.labels_tensor,
                             self.train_proc.np_embedding_matrix)

            with tf.variable_scope('optimization'):
                self.learning_rate = tf.train.exponential_decay(
                    self.hp.learning_rate,
                    self.global_step_var,
                    self.hp.decay_steps,
                    self.hp.decay_rate)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gradients = tf.gradients(
                    self.model.loss, tf.trainable_variables())
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.hp.max_grad_norm)
                grads_and_vars = tuple(
                    zip(clipped_gradients, tf.trainable_variables()))
                self.train_op = optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=self.global_step_var)

            with tf.variable_scope('metrics'):
                micro_f1, macro_f1, weighted_f1 = f1_score(
                    self.labels_tensor,
                    self.model.one_hot_prediction,
                )
                accuracy = tf.contrib.metrics.accuracy(
                    tf.cast(
                        self.model.one_hot_prediction,
                        tf.int32),
                    self.labels_tensor
                )

                self.accuracy = accuracy
                self.micro_f1 = micro_f1
                self.macro_f1 = macro_f1
                self.weighted_f1 = weighted_f1
            with tf.variable_scope('summaries'):
                tf.summary.scalar("LLoss", self.model.loss)
                tf.summary.scalar("macro_f1", macro_f1)
                tf.summary.scalar("micro_f1", micro_f1)
                tf.summary.scalar("weighted_f1", weighted_f1)
                tf.summary.scalar("accuracy", accuracy)
                tf.summary.scalar("learning_rate", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

            self.train_writer = tf.summary.FileWriter(
                str(self.hp.dir / 'tensorboard' / 'train'), self.graph)
            self.val_writer = tf.summary.FileWriter(
                str(self.hp.dir / 'tensorboard' / 'val'))

    def eval(self, tensor_name):
        return self.sess.run(self.graph.get_tensor_by_name(tensor_name))

    def initialize_iterators(self, is_val=False, inference_data=None):
        """Initializes the train and validation iterators from
        the Processers' data

            is_val (bool, optional): Defaults to False. Whether to initialize
                the validation (True) or training (False) iterator
        """
        with self.graph.as_default():
            if inference_data is not None:
                feats = inference_data
                labs = np.zeros((len(feats), self.hp.num_classes))
                bs = len(feats)
                self.sess.run(
                    self.infer_dataset_init_op,
                    feed_dict={
                        self.mode_ph: 1,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs,
                        self.shuffle_val_ph: False
                    }
                )
                print('Iterator ready to infer')
            elif is_val:
                self.val_feats = self.val_proc.features
                self.val_labs = self.val_proc.labels
                self.val_bs = len(self.val_feats)
                self.sess.run(
                    self.val_dataset_init_op,
                    feed_dict={
                        self.mode_ph: 1,
                        self.features_data_ph: self.val_feats,
                        self.labels_data_ph: self.val_labs,
                        self.batch_size_ph: self.val_bs,
                        self.shuffle_val_ph: True
                    }
                )
            else:
                self.train_feats = self.train_proc.features
                self.train_labs = self.train_proc.labels
                self.train_bs = self.hp.batch_size
                self.sess.run(
                    self.train_dataset_init_op,
                    feed_dict={
                        self.mode_ph: 0,
                        self.features_data_ph: self.train_feats,
                        self.labels_data_ph: self.train_labs,
                        self.batch_size_ph: self.train_bs,
                        self.shuffle_val_ph: True
                    }
                )

    def prepare(self):
        """Runs all steps necessary before training can start:
            * Processers should have their data ready
            * Hyperparameter is updated accordingly
            * Datasets are initialized
            * Trainer is built
        """
        if self.prepared:
            print('Already prepared, skipping.')
            return

        print('Setting Processers...')
        if self.new_procs:
            self.set_procs()

        self.hp.set_data_values(
            self.train_proc.max_doc_len,
            self.train_proc.max_sent_len,
            self.train_proc.vocab_size,
            300
        )
        self.model.hp.set_data_values(
            self.train_proc.max_doc_len,
            self.train_proc.max_sent_len,
            self.train_proc.vocab_size,
            300
        )

        print('Ok. Setting Datasets...')
        self.make_datasets()
        print('Ok. Building graph...')
        self.build()
        print('Ok. Saving hp...')
        self.hp.dump(to='json')
        print('OK.')
        self.prepared = True

    def infer(self, features, with_logits=True):
        self.initialize_iterators(inference_data=features)
        # self.sess.run(tf.assign(self.input_tensor, features))
        # if self.hp.restored:
        #     fd = {
        #         self.graph.get_tensor_by_name('mode_ph:0'): 1
        #     }
        #     prediction = self.graph.get_tensor_by_name('prediction:0')
        #     logits = self.graph.get_tensor_by_name('logits:0')
        # else:
        fd = {self.mode_ph: 1, self.shuffle_val_ph: False}
        prediction = self.model.prediction
        logits = self.model.logits

        if with_logits:
            return logits.eval(
                session=self.sess,
                feed_dict=fd
            )
        return prediction.eval(
            session=self.sess,
            feed_dict=fd
        )

    def validate(self, force=False):
        """Runs a validation step according to the current
        global_step, i.e. if step % hp.val_every == 0

        force (bool, optional): Defaults to False. Whether
            or not to force validation regarless of global_step

        Returns:
            str: Information to print
        """
        # Validate every n examples so every n/batch_size batchs
        val_every = self.hp.val_every // self.hp.batch_size

        if self.hp.global_step == 0:
            # don't validate on first step eventhough
            # self.hp.global_step% val_every == 0
            return None

        if (not force) and self.hp.global_step % val_every != 0:
            # validate if force eventhough
            # self.hp.global_step% val_every != 0
            return None

        self.initialize_iterators(is_val=True)

        s, acc, mic, mac, wei = self.sess.run(
            [
                self.summary_op,
                self.accuracy,
                self.micro_f1,
                self.macro_f1,
                self.weighted_f1
            ],
            feed_dict={
                self.mode_ph: 1,
                self.shuffle_val_ph: True
            }
        )
        self.val_writer.add_summary(s, self.hp.global_step)
        self.val_writer.flush()
        return acc, mic, mac, wei

    def epoch_string(self, epoch, loss, lr, metrics):
        val_string = ''
        if metrics is not None:
            acc, mic, mac, wei = metrics
            val_string = ' | Validation metrics: '
            val_string += 'Acc {:.4f} Mi {:.4f} Ma {:.4f} We {:.4f}\n'.format(
                acc, mic, mac, wei
            )
        epoch_string = "[{}] EPOCH {:2} / {:2} "
        epoch_string += "Step {:5} Loss: {:.4f} "
        epoch_string += "lr: {:.5f}"
        epoch_string = epoch_string.format(
            strtime(self.train_start_time),
            epoch + 1, self.hp.epochs, self.hp.global_step,
            loss, lr
        ) + val_string
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
                indexes = np.random.permutation(
                    len(self.val_proc.features)
                )[:20]
                self.ref_feats = self.val_proc.features[indexes]
                self.ref_labs = self.val_proc.labels[indexes]
            else:
                self.initialize_uninitialized()
            try:
                for epoch in range(self.hp.epochs):
                    self.initialize_iterators(is_val=False)
                    stop = False
                    while not stop:
                        try:
                            _, summary, loss, lr, gs = self.sess.run(
                                [
                                    self.train_op,
                                    self.summary_op,
                                    self.model.loss,
                                    self.learning_rate,
                                    self.global_step_var,
                                ],
                                feed_dict={
                                    self.mode_ph: 0,
                                    self.shuffle_val_ph: True
                                }
                            )
                            self.hp.global_step = gs
                            self.train_writer.add_summary(
                                summary,
                                self.hp.global_step)
                            self.train_writer.flush()

                            metrics = self.validate()
                            if gs % 100 == 0:
                                self.inferences.append(
                                    self.ref_feats
                                )

                        except tf.errors.OutOfRangeError:
                            stop = True
                        finally:
                            print(
                                self.epoch_string(epoch, loss, lr, metrics),
                                end='\r'
                            )

                # End of epochs
                if self.hp.global_step % self.hp.val_every != 0:
                    # print final validation if not done at the end
                    # of last epoch
                    print('\n[{}] Finally {}'.format(
                        strtime(self.train_start_time),
                        self.validate(force=True)
                    ))
            except KeyboardInterrupt:
                print('\nInterrupting. Save?')
                if 'y' in input('y/n : '):
                    self.save()


if __name__ == '__main__':
    # import Trainer as TR; import Hyperparameter as hyp; from importlib import reload
    # tr = TR.Trainer('HAN'); tr.train()
    pass
