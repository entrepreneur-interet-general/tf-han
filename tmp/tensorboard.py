import tensorflow as tf
from shutil import rmtree
from pathlib import Path


train_path = (Path(".") / "train").resolve().__str__()
print(train_path)
val_path = (Path(".") / "val").resolve().__str__()
rmtree(train_path, ignore_errors=True)
rmtree(val_path, ignore_errors=True)

tf.reset_default_graph()

with tf.Graph().as_default():

    sess = tf.Session()

    logits = tf.placeholder(tf.int64, [2, 3])
    labels = tf.Variable([[0, 1, 0], [1, 0, 1]])

    train_writer = tf.summary.FileWriter(train_path, tf.get_default_graph())
    val_writer = tf.summary.FileWriter(val_path)

    # create two different ops
    with tf.name_scope("train"):
        train_acc, train_acc_op = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1)
        )
        tf.summary.scalar("accuracy", train_acc, collections=["train_collection"])
    with tf.name_scope("valid"):
        valid_acc, valid_acc_op = tf.metrics.accuracy(
            labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1)
        )
        tf.summary.scalar("accuracy", valid_acc, collections=["val_collection"])

    train_sum_op = tf.summary.merge_all(key="train_collection")
    val_sum_op = tf.summary.merge_all(key="val_collection")

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        s, _ = sess.run([train_sum_op, train_acc_op], {logits: [[0, 1, 0], [1, 0, 1]]})
        train_writer.add_summary(s, i)

        if i > 0 and i % 20 == 0:
            for j in range(10):
                s, _ = sess.run(
                    [val_sum_op, valid_acc_op], {logits: [[0, 1, 0], [0, 1, 0]]}
                )
                val_writer.add_summary(s, i)
            val_writer.flush()
            stream_vars_val = [v for v in tf.local_variables() if "val" in v.name]
            sess.run(tf.variables_initializer(stream_vars_val))

    train_writer.flush()
# print(sess.run(train_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
# print(sess.run(valid_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))


# print(sess.run(valid_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
# print(sess.run(train_acc, {logits: [[0, 1, 0], [1, 0, 1]]}))
