import tensorflow as tf

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    # github.com/ematvey/hierarchical-attention-networks/blob/
    #   master/model_components.py
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            scope=scope))
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def task_specific_attention(inputs, output_size,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
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
    assert len(inputs.get_shape()
               ) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(
            name='attention_context_vector',
            shape=[output_size],
            initializer=initializer,
            dtype=tf.float32
        )
        input_projection = tf.contrib.layers.fully_connected(
            inputs, output_size,
            activation_fn=activation_fn,
            scope=scope
        )

        vector_attn = tf.reduce_sum(
            tf.multiply(input_projection, attention_context_vector),
            axis=2, keepdims=True
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


def get_graph_op(graph, and_conds=None, op='and', or_conds=None):
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
    assert op in {'and', 'or'}

    if and_conds is None:
        and_conds = ['']
    if or_conds is None:
        or_conds = ['']

    node_names = [n.name for n in graph.as_graph_def().node]

    ands = {
        n for n in node_names
        if all(
            cond in n if '!' not in cond
            else cond[1:] not in n
            for cond in and_conds
        )}

    ors = {
        n for n in node_names
        if any(
            cond in n if '!' not in cond
            else cond[1:] not in n
            for cond in or_conds
        )}

    if op == 'and':
        return [
            n for n in node_names
            if n in ands.intersection(ors)
        ]
    elif op == 'or':
        return [
            n for n in node_names
            if n in ands.union(ors)
        ]
