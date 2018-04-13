from HAN import HAN
from LogReg import LogReg
from Processer import Processer
import tensorflow as tf
from pathlib import Path


class Trainer(object):
    def __init__(self,
                 trainer_hp,
                 model_type,
                 model=None,
                 graph=None,
                 sess=None,
                 processers=None):

        self.graph = graph or tf.Graph()
        if model_type == 'LogReg':
            self.model = LogReg(trainer_hp, self.graph)
        elif model_type == 'HAN':
            self.model = HAN()
        elif model_type == 'reuse' and model:
            self.model = model
        else:
            raise ValueError('Invalid model')
        self.hp = trainer_hp

        self.new_procs = True
        if processers:
            assert isinstance(processers, list)
            if isinstance(processers[0], Processer):
                self.train_proc, self.test_proc = processers
            else:
                processers = [Path(p) for p in processers]
                self.train_proc = Processer.load(processers[0])
                self.test_proc = Processer.load(processers[1])
            self.new_procs = False

        else:
            self.train_proc = Processer()
            self.test_proc = Processer()

        self.server = None
        self.summary_op = None
        self.summary_hook = None
        self.saver_hook = None
        self.train_op = None
        self.features_data_ph = None
        self.labels_data_ph = None
        self.val_iterator = None
        self.val_dataset = None
        self.train_iterator = None
        self.train_dataset = None
        self.input_tensor = None
        self.labels_tensor = None
        self.sess = None
        self.y_pred = None

        with self.graph.as_default():
            self.global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32,
                trainable=False
            )
            self.batch_size_ph = tf.placeholder(tf.int64)
            self.val_ph = tf.placeholder(tf.bool, name='val_ph')

    def read_data(self):
        pass

    def make_datasets(self):
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

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.features_data_ph, self.labels_data_ph)
            )
            self.train_dataset = self.train_dataset.shuffle(buffer_size=100000)
            self.train_dataset = self.train_dataset.batch(self.batch_size_ph)
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
        with self.graph.as_default():
            if self.train_dataset is None:
                raise ValueError('Run make_datasets() first!')

            self.model.build(self.input_tensor, self.labels_tensor)

            self.train_op = tf.train.AdamOptimizer().minimize(
                self.model.loss,
                global_step=self.global_step
            )

            tf.summary.scalar("LLoss", self.model.loss)

            tf.summary.FileWriter(
                str(self.hp.dir / 'tensorboard'), self.graph)
            self.summary_op = tf.summary.merge_all()

    def initialize_iterators(self, is_val=False):
        with self.graph.as_default():
            if is_val:
                feats = self.test_proc.features
                labs = self.test_proc.labels
                bs = len(feats)
                self.sess.run(
                    self.val_iterator.initializer,
                    feed_dict={
                        self.val_ph: True,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs
                    }
                )
            else:
                feats = self.train_proc.features
                labs = self.train_proc.labels
                bs = self.hp.batch_size
                self.sess.run(
                    self.train_iterator.initializer,
                    feed_dict={
                        self.val_ph: False,
                        self.features_data_ph: feats,
                        self.labels_data_ph: labs,
                        self.batch_size_ph: bs
                    }
                )

    def prepare(self):
        print('Setting Processers...')
        if self.new_procs:
            self.train_proc.process(
                data_path=self.hp.train_data_path
            )
            self.test_proc.process(
                data_path=self.hp.test_data_path,
                vocabulary=self.train_proc.vocab,
                max_sent_len=self.train_proc.max_sent_len,
                max_doc_len=self.train_proc.max_doc_len
            )

        self.hp.set_data_values(
            self.train_proc.max_doc_len,
            self.train_proc.max_sent_len,
            self.train_proc.vocab_size,
            100
        )
        self.model.hp.set_data_values(
            self.train_proc.max_doc_len,
            self.train_proc.max_sent_len,
            self.train_proc.vocab_size,
            100
        )

        print('Ok. Setting Datasets...')
        self.make_datasets()
        print('Ok. Building graph...')
        self.build()

        print('Ok.')

    def train(self):
        with self.graph.as_default():
            self.prepare()
            with tf.Session() as sess:
                self.sess = sess
                tf.global_variables_initializer().run(session=self.sess)
                for epoch in range(self.hp.epochs):
                    self.initialize_iterators(is_val=False)
                    print('\n')
                    stop = False
                    while True:
                        epoch_string = '## EPOCH %d / %d' % (
                            epoch + 1, self.hp.epochs)
                        try:
                            _, loss, glob_step = self.sess.run(
                                [
                                    self.train_op,
                                    self.model.loss,
                                    self.global_step,
                                ],
                                feed_dict={self.val_ph: False}
                            )

                            epoch_string += 'Step %d Loss:  %f' % (
                                glob_step, loss)

                            if glob_step and glob_step % 20 == 0:
                                epoch_string += ' | Validation at Step %d:' % glob_step
                                self.initialize_iterators(is_val=True)
                                self.y_pred = self.sess.run(
                                    self.model.prediction,
                                    feed_dict={self.val_ph: True}
                                )
                                epoch_string += str((self.y_pred > 0.5).sum(1))
                                epoch_string += '\n'

                        except tf.errors.OutOfRangeError:
                            stop = True
                        finally:
                            print(epoch_string, end='\r')
                            if stop:
                                break
