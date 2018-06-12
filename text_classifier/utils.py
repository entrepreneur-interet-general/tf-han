from importlib import reload
from types import ModuleType

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, variable_scope

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

split_doc_token = "|&|"
padding_token = "<PAD>"


def _reload(module, name):
    reload(module)
    reload(module)
    print("Reloading ", module)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType:
            if name in attribute.__name__:
                _reload(attribute, name)


def rreload(module):
    """Recursively reload modules."""
    name = module.__name__
    _reload(module, name)


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths, scope=None):
    # github.com/ematvey/hierarchical-attention-networks/blob/
    #   master/model_components.py
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        (
            (fw_outputs, bw_outputs),
            (fw_state, bw_state),
        ) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs_embedded,
            sequence_length=input_lengths,
            dtype=tf.float32,
            swap_memory=True,
            scope=scope,
        )
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name="bidirectional_concat_c"
                )
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name="bidirectional_concat_h"
                )
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1, name="bidirectional_concat")
                return state
            elif (
                isinstance(fw_state, tuple)
                and isinstance(bw_state, tuple)
                and len(fw_state) == len(bw_state)
            ):
                # multilayer
                state = tuple(
                    concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state)
                )
                return state

            else:
                raise ValueError("unknown state type: {}".format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def task_specific_attention(
    inputs,
    output_size,
    initializer=tf.contrib.layers.xavier_initializer(),
    activation_fn=tf.tanh,
    scope=None,
):
    # github.com/ematvey/hierarchical-attention-networks/blob/
    #   master/model_components.py
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) axisension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or "attention") as scope:
        attention_context_vector = tf.get_variable(
            name="attention_context_vector",
            shape=[output_size],
            initializer=initializer,
            dtype=tf.float32,
        )
        input_projection = tf.contrib.layers.fully_connected(
            inputs, output_size, activation_fn=activation_fn, scope=scope
        )

        vector_attn = tf.reduce_sum(
            tf.multiply(input_projection, attention_context_vector),
            axis=2,
            keepdims=True,
        )
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs


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


def is_prop(obj, attr):
    return isinstance(getattr(type(obj), attr, None), property)


def get_graph_op(graph, and_conds=None, op="and", or_conds=None):
    """Selects nodes' names in the graph if:
    * The name contains all items in and_conds
    * OR/AND depending on op
    * The name contains any item in or_conds

    Condition starting with a "!" are negated.
    Returns all ops if no optional arguments is given.

    Args:
        graph (tf.Graph): The graph containing sought tensors
        and_conds (list(str)), optional): Defaults to None.
            "and" conditions
        op (str, optional): Defaults to 'and'. 
            How to link the and_conds and or_conds:
            with an 'and' or an 'or'
        or_conds (list(str), optional): Defaults to None.
            "or conditions"

    Returns:
        list(str): list of relevant tensor names
    """
    assert op in {"and", "or"}

    if and_conds is None:
        and_conds = [""]
    if or_conds is None:
        or_conds = [""]

    node_names = [n.name for n in graph.as_graph_def().node]

    ands = {
        n
        for n in node_names
        if all(
            cond in n if "!" not in cond else cond[1:] not in n for cond in and_conds
        )
    }

    ors = {
        n
        for n in node_names
        if any(cond in n if "!" not in cond else cond[1:] not in n for cond in or_conds)
    }

    if op == "and":
        return [n for n in node_names if n in ands.intersection(ors)]

    return [n for n in node_names if n in ands.union(ors)]


def extract_words(string):
    # Split string
    out = tf.string_split(string, delimiter=" ")
    # Convert to Dense tensor, filling with default value
    # Which is the padding token in this case
    out = tf.sparse_tensor_to_dense(out, default_value=padding_token)
    return out


def extract_sents(string):
    # Split the document line into sentences
    return tf.string_split([string], split_doc_token).values


def one_hot_label(string_label):
    # convert a string to a one-hot encoded label
    # '3' > [0, 0, 0, 1, 0]
    return tf.one_hot(tf.string_to_number(string_label, out_type=tf.int64), 5, 1, 0)


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


def streaming_counts(_y_true, _y_pred, num_classes):
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

    y_true = tf.cast(_y_true, tf.int64)
    y_pred = tf.cast(_y_pred, tf.int64)

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

    # computing the micro f1 score
    with tf.variable_scope("micro"):
        prec_mic = tp_mic / (tp_mic + fp_mic)
        rec_mic = tp_mic / (tp_mic + fn_mic)
        f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
        f1_mic = tf.reduce_mean(f1_mic)

    # computing the macro and wieghted f1 score
    with tf.variable_scope("macro"):
        prec_mac = tp_mac / (tp_mac + fp_mac)
        rec_mac = tp_mac / (tp_mac + fn_mac)
        _f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
        f1_mac = tf.reduce_mean(_f1_mac)
    with tf.variable_scope("weighted"):
        # normalize weights
        weights /= tf.reduce_sum(weights)
        f1_wei = tf.reduce_sum(f1_mac * weights)

    return f1_mic, f1_mac, f1_wei
