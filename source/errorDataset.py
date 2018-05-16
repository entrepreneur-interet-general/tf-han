import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from utils import get_graph_op


def model(graph, input_tensor):
    cell = tf.nn.rnn_cell.GRUCell(10)
    with graph.as_default():
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                            cell_bw=cell,
                                            inputs=input_tensor,
                                            sequence_length=[10] * 32,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=None))
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        mean = tf.reduce_mean(outputs, axis=1)
        dense = tf.layers.dense(
            mean, 5, activation=None)

        return dense


if __name__ == '__main__':

    np.random.seed(0)
    features = np.random.randn(64, 10, 30)
    labels = np.eye(5)[np.random.randint(0, 5, (64,))]

    graph1 = tf.Graph()
    with graph1.as_default():
        tf.set_random_seed(0)
        # Placeholders
        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
        features_data_ph = tf.placeholder(
            tf.float32,
            [None, None, 30],
            'features_data_ph'
        )
        labels_data_ph = tf.placeholder(
            tf.int32,
            [None, 5],
            'labels_data_ph'
        )
        # Dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (features_data_ph, labels_data_ph))
        dataset = dataset.shuffle(
            buffer_size=100000)
        dataset = dataset.batch(
            batch_size_ph)
        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(
            dataset, name='dataset_init')
        input_tensor, labels_tensor = iterator.get_next()

        output = model(graph1, input_tensor)

        with tf.Session(graph=graph1) as sess:

            tf.global_variables_initializer().run(session=sess)
            for epoch in range(2):
                batch = 0
                sess.run(
                    dataset_init_op,
                    feed_dict={
                        features_data_ph: features,
                        labels_data_ph: labels,
                        batch_size_ph: 32
                    }
                )
                while True:
                    try:
                        print(epoch, batch, sess.run(output)[0])
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break
            cwd = os.getcwd()
            path = os.path.join(cwd, 'test_simple')
            shutil.rmtree(path, ignore_errors=True)
            inputs_dict = {
                "batch_size_ph": batch_size_ph,
                "features_data_ph": features_data_ph,
                "labels_data_ph": labels_data_ph
            }
            outputs_dict = {
                "output": output
            }
            tf.saved_model.simple_save(
                sess, path, inputs_dict, outputs_dict
            )
    # Restoring
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                path
            )
            l = get_graph_op(graph2, ['!bidir'])
            dataset_init_op = graph2.get_operation_by_name('dataset_init')
            logits = graph2.get_tensor_by_name('dense/BiasAdd:0')
            labels_data_ph = graph2.get_tensor_by_name('labels_data_ph:0')
            features_data_ph = graph2.get_tensor_by_name('features_data_ph:0')
            batch_size_ph = graph2.get_tensor_by_name('batch_size_ph:0')

            sess.run(
                dataset_init_op,
                feed_dict={
                    features_data_ph: features,
                    labels_data_ph: labels,
                    batch_size_ph: 32
                }
            )
            print(sess.run(logits)[0])
