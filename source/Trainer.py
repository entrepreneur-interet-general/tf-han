import tensorflow as tf


class Trainer(object):
    def __init__(self, model, trainer_hp):
        self.model = model
        self.hp = trainer_hp
        
        self.global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            dtype=tf.int32,
            trainable=False
        )

        self.server = None
        self.summary_op = None
        self.summary_hook = None
        self.saver_hook = None
        self.train_op = None

    def read_data(self):
        pass

    def build(self):

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

    def train(self):
        with tf.train.MonitoredTrainingSession(
                master=self.server.target,
                hooks=[self.summary_hook],
                is_chief=True) as sess:
            tf.train.start_queue_runners(sess=sess)
            for _ in range(2):
                _, loss, glob_step = sess.run([
                    self.train_op,
                    self.model.loss,
                    self.global_step
                ])
                print(loss, glob_step)
