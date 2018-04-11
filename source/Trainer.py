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
        self.dataset = None
        self.features_data_ph = None
        self.labels_data_ph = None
        self.iterator = None
        self.dataset = None
        self.input_tensor = None
        self.labels_tensor = None
        self.sess = None

        with self.graph.as_default():
            self.global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int32,
                trainable=False
            )
            self.batch_size_ph = tf.placeholder(tf.int64)

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

            self.dataset = tf.data.Dataset.from_tensor_slices(
                (self.features_data_ph, self.labels_data_ph)
            )
            self.dataset = self.dataset.shuffle(buffer_size=100000)
            self.dataset = self.dataset.batch(self.batch_size_ph)
            self.iterator = self.dataset.make_initializable_iterator()

            self.input_tensor, self.labels_tensor = self.iterator.get_next()

    def build(self):
        with self.graph.as_default():
            if self.dataset is None:
                raise ValueError('Run make_datasets() first!')

            self.model.build(self.input_tensor, self.labels_tensor)

            self.train_op = tf.train.AdamOptimizer().minimize(
                self.model.loss,
                global_step=self.global_step
            )

            tf.summary.scalar("loss", self.model.loss)

            self.server = tf.train.Server.create_local_server()
            self.summary_op = tf.summary.merge_all()
            self.summary_hook = tf.train.SummarySaverHook(
                save_secs=self.hp.summary_secs,
                output_dir=str(self.hp.dir),
                summary_op=self.summary_op)
            self.saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=self.hp.dir,
                save_steps=self.hp.save_steps,
            )

    def initialize_iterators(self, is_test=False):
        with self.graph.as_default():
            if is_test:
                feats = self.test_proc.features
                labs = self.test_proc.labels
                bs = len(feats)
            else:
                feats = self.train_proc.features
                labs = self.train_proc.labels
                bs = self.hp.batch_size

            self.sess.run(
                self.iterator.initializer,
                feed_dict={
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
                max_sent_len=self.train_proc.max_sent_len
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
            with tf.train.MonitoredTrainingSession(
                    hooks=[self.summary_hook, self.saver_hook]) as self.sess:

                self.initialize_iterators(is_test=False)
                for epoch in range(self.hp.epochs):
                    print('## EPOCH %d' % epoch)
                    while not self.sess.should_stop():

                        _, loss, glob_step = self.sess.run([
                            self.train_op,
                            self.model.loss,
                            self.global_step
                        ])

                        print(loss)

                        if glob_step and glob_step % 100 == 0:
                            print('> Step %d' % glob_step)
                            self.initialize_iterators(is_test=True)
                            y_pred = self.sess.run(
                                self.model.prediction
                            )
                            self.initialize_iterators(is_test=False)
