import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from sklearn.metrics import f1_score


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections.
    from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/metrics_impl.py
    """

    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES],
        validate_shape=validate_shape,
        name=name,
    )


def streaming_counts(y_true, y_pred, num_classes):
    """Computes the TP, FP and FN counts for the micro and macro f1 scores.
    The weighted f1 score can be inferred from the macro f1 score provided
    we compute the weights also.

    This function also defines the update ops to these counts
    
    Args:
        y_true (Tensor): 2D Tensor representing the target labels
        y_pred (Tensor): 2D Tensor representing the predicted labels
        num_classes (int): number of possible classes

    Returns:
        tuple: the first element in the tuple is itself a tuple grouping the counts,
        the second element is the grouped update op.
    """

    # Weights for the weighted f1 score
    weights = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="weights"
    )
    # Counts for the macro f1 score
    tp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="tp_mac"
    )
    fp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fp_mac"
    )
    fn_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fn_mac"
    )
    # Counts for the micro f1 score
    tp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
    )
    fp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )
    fn_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fn_mic"
    )

    # Update ops, as in the previous section:
    #   - Update ops for the macro f1 score
    up_tp_mac = tf.assign_add(tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
    up_fp_mac = tf.assign_add(fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
    up_fn_mac = tf.assign_add(fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))

    #   - Update ops for the micro f1 score
    up_tp_mic = tf.assign_add(tp_mic, tf.count_nonzero(y_pred * y_true, axis=None))
    up_fp_mic = tf.assign_add(
        fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    )
    up_fn_mic = tf.assign_add(
        fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None)
    )
    # Update op for the weights, just summing
    up_weights = tf.assign_add(weights, tf.reduce_sum(y_true, axis=0))

    # Grouping values
    counts = (tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights)
    updates = tf.group(
        up_tp_mic, up_fp_mic, up_fn_mic, up_tp_mac, up_fp_mac, up_fn_mac, up_weights
    )

    return counts, updates


def streaming_f1(y_true, y_pred, num_classes):
    """Compute and update the F1 scores given target labels
    and predicted labels
    
    Args:
        y_true (Tensor): 2D one-hot Tensor of the target labels.
            Possibly several ones for multiple labels
        y_pred (Tensor): 2D one-hot Tensor of the predicted labels.
            Possibly several ones for multiple labels
        num_classes (int): Number of possible classes labels can take
    
    Returns:
        tuple: f1s as tuple of three tensors: micro macro and weighted F1,
            second element is the group of updates to counts making the F1s
    """
    counts, updates = streaming_counts(y_true, y_pred, num_classes)
    f1s = streaming_f1_from_counts(counts)
    return f1s, updates


def streaming_f1_from_counts(counts):
    """Computes the f1 scores from the TP, FP and FN counts
    
    Args:
        counts (tuple): macro and micro counts, and weights in the end
    
    Returns:
        tuple(Tensor): The 3 tensors representing the micro, macro and weighted
            f1 score
    """
    # unpacking values
    tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights = counts

    # normalize weights
    weights /= tf.reduce_sum(weights)

    # computing the micro f1 score
    prec_mic = tp_mic / (tp_mic + fp_mic)
    rec_mic = tp_mic / (tp_mic + fn_mic)
    f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
    f1_mic = tf.reduce_mean(f1_mic)

    # computing the macro and wieghted f1 score
    prec_mac = tp_mac / (tp_mac + fp_mac)
    rec_mac = tp_mac / (tp_mac + fn_mac)
    f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
    f1_wei = tf.reduce_sum(f1_mac * weights)
    f1_mac = tf.reduce_mean(f1_mac)

    return f1_mic, f1_mac, f1_wei


def tf_f1_score(y_true, y_pred):
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


def alter_data(_data):
    """Adds noise to the data to simulate predictions.
    Each label for each sample has 20% chance of being flipped
    
    Args:
        _data (np.array): true values to perturb
    
    Returns:
        np.array: predictions
    """
    data = _data.copy()
    new_data = []
    for d in data:
        for i, l in enumerate(d):
            if np.random.rand() < 0.2:
                d[i] = (d[i] + 1) % 2
        new_data.append(d)
    return np.array(new_data)


def get_data(multilabel=True):
    """Generate random multilabel data:

    y_true and y_pred are one-hot arrays, but since it's a multi-label setting,
        there may be several `1` per line.

    Returns:
        tuple: y_true, y_pred
    """
    # Number of different classes
    num_classes = 10
    classes = list(range(num_classes))
    # Numberof samples in synthetic dataset
    examples = 10000
    # Max number of labels per sample. Minimum is 1
    max_labels = 5 if multilabel else 1
    class_probabilities = np.array(
        list(6 * np.exp(-i * 5 / num_classes) + 1 for i in range(num_classes))
    )
    class_probabilities /= class_probabilities.sum()
    labels = [
        np.random.choice(
            classes,
            # number of labels for this sample
            size=np.random.randint(1, max_labels + 1),
            p=class_probabilities,  # Probability of drawing each class
            replace=False,  # A class can only occure once
        )
        for _ in range(examples)  # Do it `examples` times
    ]
    y_true = np.zeros((examples, num_classes)).astype(np.int64)
    for i, l in enumerate(labels):
        y_true[i][l] = 1
    y_pred = alter_data(y_true)
    return y_true, y_pred


if __name__ == "__main__":
    np.random.seed(0)
    y_true, y_pred = get_data(False)
    num_classes = y_true.shape[-1]

    bs = 100

    t = tf.placeholder(tf.int64, [None, None], "y_true")
    p = tf.placeholder(tf.int64, [None, None], "y_pred")
    tf_f1 = tf_f1_score(t, p)
    counts, update = streaming_counts(t, p, num_classes)
    streamed_f1 = streaming_f1_from_counts(counts)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()

        mic, mac, wei = sess.run(tf_f1, feed_dict={t: y_true, p: y_pred})
        print("{:40}".format("\nTotal, overall f1 scores: "), mic, mac, wei)

        for i in range(len(y_true) // bs):
            y_t = y_true[i * bs : (i + 1) * bs].astype(np.int64)
            y_p = y_pred[i * bs : (i + 1) * bs].astype(np.int64)
            _ = sess.run(update, feed_dict={t: y_t, p: y_p})
        mic, mac, wei = [f.eval() for f in streamed_f1]
        print("{:40}".format("\nStreamed, batch-wise f1 scores:"), mic, mac, wei)

    mic = f1_score(y_true, y_pred, average="micro")
    mac = f1_score(y_true, y_pred, average="macro")
    wei = f1_score(y_true, y_pred, average="weighted")
    print("{:40}".format("\nFor reference, scikit f1 scores:"), mic, mac, wei)
