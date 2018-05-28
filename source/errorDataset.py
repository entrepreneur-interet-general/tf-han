import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


def model(graph, input_tensor):
    """Create the model which consists of
    a bidirectional rnn (GRU(10)) folowed by a dense classifier

    Args:
        graph (tf.Graph): Tensors' graph
        input_tensor (tf.Tensor): Tensor fed as input to the model

    Returns:
        tf.Tensor: the model's output layer Tensor
    """
    cell = tf.nn.rnn_cell.GRUCell(10)
    with graph.as_default():
        (
            (fw_outputs, bw_outputs),
            (fw_state, bw_state),
        ) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=input_tensor,
            sequence_length=[10] * 32,
            dtype=tf.float32,
            swap_memory=True,
            scope=None,
        )
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        mean = tf.reduce_mean(outputs, axis=1)
        dense = tf.layers.dense(mean, 5, activation=None)

        return dense


def get_opt_op(graph, logits, labels_tensor):
    """Create optimization operation from model's logits and labels

    Args:
        graph (tf.Graph): Tensors' graph
        logits (tf.Tensor): The model's output without activation
        labels_tensor (tf.Tensor): Target labels

    Returns:
        tf.Operation: the operation performing a stem of Adam optimizer
    """
    with graph.as_default():
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels_tensor, name="xent"
                ),
                name="mean-xent",
            )
        with tf.variable_scope("optimizer"):
            opt_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
        return opt_op


if __name__ == "__main__":
    # Set random seed for reproductability
    # and create synthetic data
    np.random.seed(0)
    features = np.random.randn(64, 10, 30)
    labels = np.eye(5)[np.random.randint(0, 5, (64,))]

    graph1 = tf.Graph()
    with graph1.as_default():
        # Random seed for reproductability
        tf.set_random_seed(0)
        # Placeholders
        batch_size_ph = tf.placeholder(tf.int64, name="batch_size_ph")
        features_data_ph = tf.placeholder(
            tf.float32, [None, None, 30], "features_data_ph"
        )
        labels_data_ph = tf.placeholder(tf.int32, [None, 5], "labels_data_ph")
        # Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph))
        dataset = dataset.batch(batch_size_ph)
        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes
        )
        dataset_init_op = iterator.make_initializer(dataset, name="dataset_init")
        input_tensor, labels_tensor = iterator.get_next()

        # Model
        logits = model(graph1, input_tensor)
        # Optimization
        opt_op = get_opt_op(graph1, logits, labels_tensor)

        with tf.Session(graph=graph1) as sess:
            # Initialize variables
            tf.global_variables_initializer().run(session=sess)
            for epoch in range(3):
                batch = 0
                # Initialize dataset
                # (could feed epochs in Dataset.repeat(epochs))
                sess.run(
                    dataset_init_op,
                    feed_dict={
                        features_data_ph: features,
                        labels_data_ph: labels,
                        batch_size_ph: 32,
                    },
                )
                values = []
                while True:
                    try:
                        if epoch < 2:
                            # Training
                            _, value = sess.run([opt_op, logits])
                            print(epoch, batch, value[0])
                            batch += 1
                        else:
                            # Final inference
                            values.append(sess.run(logits))
                            print(epoch, batch, values[-1][0])
                            batch += 1
                    except tf.errors.OutOfRangeError:
                        break
            # Save model state
            cwd = os.getcwd()
            path = os.path.join(cwd, "test_simple")
            shutil.rmtree(path, ignore_errors=True)
            inputs_dict = {
                "batch_size_ph": batch_size_ph,
                "features_data_ph": features_data_ph,
                "labels_data_ph": labels_data_ph,
            }
            outputs_dict = {"logits": logits}
            tf.saved_model.simple_save(sess, path, inputs_dict, outputs_dict)
    # Restoring
    graph2 = tf.Graph()
    with graph2.as_default():
        with tf.Session(graph=graph2) as sess:
            # Restore saved values
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], path)
            # Get restored placeholders
            labels_data_ph = graph2.get_tensor_by_name("labels_data_ph:0")
            features_data_ph = graph2.get_tensor_by_name("features_data_ph:0")
            batch_size_ph = graph2.get_tensor_by_name("batch_size_ph:0")
            # Get restored model output
            restored_logits = graph2.get_tensor_by_name("dense/BiasAdd:0")
            # Get dataset initializing operation
            dataset_init_op = graph2.get_operation_by_name("dataset_init")

            # Initialize restored dataset
            sess.run(
                dataset_init_op,
                feed_dict={
                    features_data_ph: features,
                    labels_data_ph: labels,
                    batch_size_ph: 32,
                },
            )
            # Compute inference for both batches in dataset
            restored_values = []
            for i in range(2):
                restored_values.append(sess.run(restored_logits))
                print(restored_values[i][0])
    # Check if original inference and restored inference are equal
    valid = all((v == rv).all() for v, rv in zip(values, restored_values))
    print("\nInferences match: ", valid)
