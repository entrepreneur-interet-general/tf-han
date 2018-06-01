import tensorflow as tf
import numpy as np
from pathlib import Path

split_string_token = "|&|"
count = 0


def extract_words(string):
    # Split characters
    out = tf.string_split(string, delimiter=" ")
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value="<PAD>")
    return out


def get_vocab(_file):
    with Path(_file).resolve().open("r") as f:
        lines = f.readlines()
    return {w[:-1]: i for i, w in enumerate(lines)}


def get_docs(_file):
    with Path(_file).open("r") as f:
        lines = f.readlines()
    return [[s.split(" ") for s in l.split("|&|")] for l in lines]


def sample(docs, batch_size):
    return [docs[i] for i in np.random.permutation(len(docs))[:batch_size]]


def lookup(docs, vocab):
    z = np.zeros(
        (len(docs), max(len(d) for d in docs), max(len(s) for d in docs for s in d))
    )
    for i, d in enumerate(docs):
        for j, s in enumerate(d):
            for k, w in enumerate(s):
                z[i, j, k] = vocab.get(w, 0)
    return z


def get_example(docs, batch_size, vocab):
    s = sample(docs, batch_size)
    return lookup(s, vocab)


def get_tf_sample(sess, elements, iterator_init_op):
    return sess.run(elements)[0]


if __name__ == "__main__":
    try:
        sess.close()
    except NameError:
        pass

    num_threads = 3

    tf.reset_default_graph()
    padded_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None]))
    padding_values = (np.int64(0), 0)

    words = tf.contrib.lookup.index_table_from_file(
        "../data/yelp/tf-prepared/sample_0001_val_01/words.txt", num_oov_buckets=1
    )
    documents_dataset = tf.data.TextLineDataset(
        "../data/yelp/tf-prepared/sample_0001_val_01/documents.txt"
    )
    labels = tf.data.TextLineDataset(
        "../data/yelp/tf-prepared/sample_0001_val_01/labels.txt"
    ).map(
        lambda lab: tf.one_hot(
            tf.string_to_number(lab, out_type=tf.int64, name=None), 5, 1, 0
        )
    )
    documents_dataset = documents_dataset.map(
        lambda string: tf.string_split([string], split_string_token).values,
        num_parallel_calls=num_threads,
    )
    documents_dataset = documents_dataset.map(
        extract_words, num_parallel_calls=num_threads
    )
    documents_dataset = documents_dataset.map(lambda tokens: words.lookup(tokens))
    dataset = tf.data.Dataset.zip((documents_dataset, labels))
    dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(32, padded_shapes, padding_values)
    dataset = dataset.prefetch(10)
    # documents_dataset = documents_dataset.

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes
    )
    iterator_init_op = iterator.make_initializer(dataset, name="train_dataset_init_op")

    elements = iterator.get_next()

    sess = tf.Session()
    sess.run([iterator_init_op, tf.tables_initializer()])
    el = sess.run(elements)
    docs = np.array(el[0])
    print(docs.shape)
    # count = 0
    # while docs.shape[-1] == 30:
    #     docs, labels = sess.run(elements)
    #     count += 1
    # print(count)

    vocab = get_vocab("../data/yelp/tf-prepared/sample_0001_val_01/words.txt")
    docs = get_docs("../data/yelp/tf-prepared/sample_0001_val_01/documents.txt")
