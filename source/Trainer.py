from HAN import HAN
from LogReg import LogReg
from Processer import Processer
import tensorflow as tf
from pathlib import Path
from time import time
import shutil


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
    def f1_score(y_true, y_pred):
        """Computes 3 different f1 scores, micro macro
        weighted.
        micro: f1 score accross the classes, as 1
        macro: mean of f1 scores per class
        weighted: weighted average of f1 scores per class,
                  weighted from the support of each class


        Args:
            y_true (Tensor): labels, with shape (batch, num_classes)
            y_pred (Tensor): model's predictions, same shape as y_true

        Returns:
            tupe(Tensor): (micro, macro, weighted)
                          tuple of the computed f1 scores
        """

        f1s = [0, 0, 0]

        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        for i, axis in enumerate([None, 0]):
            TP = tf.count_nonzero(y_pred * y_true, axis=axis)
            FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
            FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)

            f1s[i] = tf.reduce_mean(f1)

        weights = tf.reduce_sum(y_true, axis=0)
        weights /= tf.reduce_sum(weights)

        f1s[2] = tf.reduce_sum(f1 * weights)

        micro, macro, weighted = f1s
        return micro, macro, weighted

    def __init__(self,
                 trainer_hp,
                 model_type,
                 model=None,
                 graph=None,
                 sess=None,
                 processers=None):
        """Creates a Trainer.

        Args:
            trainer_hp (Hyperparameter): the trainer's params
            model_type (str): can be a known model or "reuse"
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
        self.sess = sess or tf.Session(graph=self.graph)
        self.prepared = False

        if model_type == 'LogReg':
            self.model = LogReg(trainer_hp, self.graph)
        elif model_type == 'HAN':
            self.model = HAN(trainer_hp, self.graph)
        elif model_type == 'reuse' and model:
            self.model = model
        else:
            raise ValueError('Invalid model')
        self.hp = trainer_hp

        self.new_procs = True
        if processers:
            assert isinstance(processers, list)
            assert len(processers) == 2

            if isinstance(processers[0], proc.Processer):
                self.train_proc, self.val_proc = processers
            else:
                processers = [Path(p) for p in processers]
                self.train_proc = Processer.load(processers[0])
                self.val_proc = Processer.load(processers[1])
            self.new_procs = False

        else:
            self.train_proc = Processer()
            self.val_proc = Processer()

        self.server = None
        self.summary_op = None
        self.train_op = None
        self.features_data_ph = None
        self.labels_data_ph = None
        self.val_iterator = None
        self.val_dataset = None
        self.train_iterator = None
        self.train_dataset = None
        self.input_tensor = None
        self.labels_tensor = None
        self.y_pred = None
        self.train_writer = None
        self.val_writer = None
        self.global_step_var = None
        self.global_step = None
        self.train_start_time = None

        with self.graph.as_default():
            self.global_step_var = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32,
                trainable=False
            )
            self.batch_size_ph = tf.placeholder(tf.int64)
            self.val_ph = tf.placeholder(tf.bool, name='val_ph')
            self.keep_prob_dropout_ph = tf.placeholder(
                tf.float32, name='keep_prob_dropout_ph'
            )
            self.model.val_ph = self.val_ph
            self.model.keep_prob_dropout_ph = self.keep_prob_dropout_ph

    def delete(self, ask=True):
        """Delete the trainer's directory after asking for confiirmation

            ask (bool, optional): Defaults to True.
            Whether to ask for confirmation
        """
        if not ask or 'y' in input('Are you sure? (y/n)'):
            shutil.rmtree(self.hp.dir)

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
        Conditioned on the value of val_ph, the input and label tensors come
        from either the validation (True) or training (False) set.
        """

        with self.graph.as_default():
            with tf.name_scope('dataset'):
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

                self.train_dataset = tf.data.Dataset.from_tensor_slices(
                    (self.features_data_ph, self.labels_data_ph)
                )
                self.train_dataset = self.train_dataset.shuffle(
                    buffer_size=100000)
                self.train_dataset = self.train_dataset.batch(
                    self.batch_size_ph)
                self.train_iterator = \
                    self.train_dataset.make_initializable_iterator()

                self.val_dataset = tf.data.Dataset.from_tensor_slices(
                    (self.features_data_ph, self.labels_data_ph)
                )
                self.val_dataset = self.val_dataset.shuffle(buffer_size=100000)
                self.val_dataset = self.val_dataset.batch(self.batch_size_ph)
                self.val_iterator = \
                    self.val_dataset.make_initializable_iterator()

                self.input_tensor, self.labels_tensor = tf.cond(
                    self.val_ph,
                    self.val_iterator.get_next,
                    self.train_iterator.get_next
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

            with tf.name_scope('optimizer'):
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.hp.learning_rate).minimize(
                    self.model.loss,
                    global_step=self.global_step_var
                )

            with tf.name_scope('metrics'):
                micro_f1, macro_f1, weighted_f1 = self.f1_score(
                    self.labels_tensor,
                    self.model.prediction,
                )
                accuracy = tf.contrib.metrics.accuracy(
                    self.model.prediction,
                    self.labels_tensor
                )
            with tf.name_scope('summaries'):
                tf.summary.scalar("LLoss", self.model.loss)
                tf.summary.scalar("macro_f1", macro_f1)
                tf.summary.scalar("micro_f1", micro_f1)
                tf.summary.scalar("weighted_f1", weighted_f1)
                tf.summary.scalar("accuracy", accuracy)
                self.summary_op = tf.summary.merge_all()

            self.train_writer = tf.summary.FileWriter(
                str(self.hp.dir / 'tensorboard' / 'train'), self.graph)
            self.val_writer = tf.summary.FileWriter(
                str(self.hp.dir / 'tensorboard' / 'val'))

    def initialize_iterators(self, is_val=False):
        """Initializes the train and validation iterators from
        the Processers' data

            is_val (bool, optional): Defaults to False. Whether to initialize
                the validation (True) or training (False) iterator
        """
        with self.graph.as_default():
            if is_val:
                feats = self.val_proc.features
                labs = self.val_proc.labels
                bs = len(feats)
                self.sess.run(
                    self.val_iterator.initializer,
                    feed_dict={
                        self.val_ph: True,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs,
                        self.keep_prob_dropout_ph: 1.0
                    }
                )
            else:
                feats = self.train_proc.features
                labs = self.train_proc.labels
                bs = self.hp.val_batch_size
                self.sess.run(
                    self.train_iterator.initializer,
                    feed_dict={
                        self.val_ph: False,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs,
                        self.keep_prob_dropout_ph: self.hp.dropout
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
        print('Ok.')
        self.prepared = True

    def validate(self, force=False):
        """Runs a validation step according to the current
        global_step, i.e. if step % hp.val_every == 0

        force (bool, optional): Defaults to False. Whether
            or not to force validation regarless of global_step

        Returns:
            str: Information to print
        """
        if self.global_step == 0:
            # don't validate on first step eventhough
            # self.global_step % self.hp.val_every == 0
            return ''

        if (not force) and self.global_step % self.hp.val_every != 0:
            # validate if force eventhough
            # self.global_step % self.hp.val_every != 0
            return ''

        self.initialize_iterators(is_val=True)

        summary, self.y_pred = self.sess.run(
            [
                self.summary_op,
                self.model.prediction
            ],
            feed_dict={
                self.val_ph: True,
                self.keep_prob_dropout_ph: 1.0
            }
        )
        self.val_writer.add_summary(summary, self.global_step)
        self.val_writer.flush()
        val_string = ' | Validation at Step {}: {}\n'.format(
            self.global_step, str((self.y_pred > 0.5).sum(1))
        )
        return val_string

    def train(self):
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
        epoch_string = ""
        self.train_start_time = time()
        with self.graph.as_default():
            self.prepare()
            tf.global_variables_initializer().run(session=self.sess)
            for epoch in range(self.hp.epochs):
                self.initialize_iterators(is_val=False)
                stop = False
                while True:
                    try:
                        _, summary, loss, gs = self.sess.run(
                            [
                                self.train_op,
                                self.summary_op,
                                self.model.loss,
                                self.global_step_var,
                            ],
                            feed_dict={
                                self.val_ph: False,
                                self.keep_prob_dropout_ph: self.hp.dropout
                            }
                        )
                        self.global_step = gs
                        self.train_writer.add_summary(
                            summary,
                            self.global_step
                        )
                        self.train_writer.flush()

                        val_string = self.validate()
                        epoch_string = "[{}] EPOCH {:3} / {:3} "
                        epoch_string += "Step {:5} Loss: {:.3f} {}"
                        epoch_string = epoch_string.format(
                            strtime(self.train_start_time),
                            epoch + 1, self.hp.epochs, self.global_step,
                            loss, val_string
                        )

                    except tf.errors.OutOfRangeError:
                        stop = True
                    finally:
                        print(epoch_string, end='\r')
                        if stop:
                            break
            # End of epochs
            if self.global_step % self.hp.val_every != 0:
                # print final validation if not done at the end
                # of last epoch
                print('\n[{}] Finally {}'.format(
                    strtime(self.train_start_time),
                    self.validate(True)
                ))


if __name__ == '__main__':
    import Processer as proc
    import Hyperparameter as hyp
    import Model as mod
    import LogReg as lr
    import HAN as han

    from importlib import reload

    reload(proc)
    reload(hyp)
    reload(mod)
    reload(lr)
    reload(han)

    f = '/Users/victor/Documents/Tracfin/dev/han/data/embeddings/'
    f += 'glove.840B.300d.txt'

    hp = hyp.HP(
        multilabel=False,
        batch_size=32,
        learning_rate=5e-4,
        cell_size=100,
        epochs=5,
        val_every=1000,
        val_batch_size=1000,
        embedding_file=f,
        max_words=1e6,
        num_classes=5,
        train_data_path='/Users/victor/Documents/Tracfin/dev/han/data/yelp/sample_001_train_07.json',
        val_data_path='/Users/victor/Documents/Tracfin/dev/han/data/yelp/sample_001_val_01.json')
    print('Resetting default graph...')
    tf.reset_default_graph()
    print('Ok.')
    # procs = [
    #     proc.Processer.load('../checkpoints/v1/2018-04-15/train_proc.pkl'),
    #     proc.Processer.load('../checkpoints/v1/2018-04-15/val_proc.pkl')
    # ]

    # trainer = Trainer(
    #     hp, 'LogReg'
    # )

    trainer = Trainer(hp, 'HAN')
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass

    if 'y' in input('\nDone. delete?'):
        trainer.delete(False)
