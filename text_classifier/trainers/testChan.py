import tensorflow as tf
import numpy as np

pad_word = "/"
split_doc_token = "|&|"


def one_hot_multi_label(string_one_hot):
    """Converts a one-hot string multilabel to a tf.int64 tensor:
    "1.0, 0.0, 1.0, 0.0, 0.0" -> [1, 0, 1, 0, 0]
    
    Args:
        string_one_hot (Tensor): string label to convert
    
    Returns:
        Tensor: tf.int64 Tensor of one-hot encoded labels
    """
    vals = tf.string_split([string_one_hot], split_label_token).values
    numbs = tf.string_to_number(vals, out_type=tf.float64)
    return tf.cast(numbs, tf.int64)


def extract_chars(sentences):
    # sentences = list( list(word = str) = sentence) = document
    words = tf.reshape(sentences, (-1,))
    out = tf.string_split(words, delimiter="")
    out = tf.sparse_tensor_to_dense(out, default_value=pad_word)
    out = tf.reshape(out, (tf.shape(sentences)[0], tf.shape(sentences)[1], -1))
    print(out)
    return out


def extract_words(document):
    # document = list(sentences = str)
    # Split characters
    out = tf.string_split(document, delimiter=" ")
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value=pad_word)
    return out


def extract_sents(document_string):
    # Split the document line into sentences
    return tf.string_split([document_string], split_doc_token).values


def preprocess_dataset(doc_ds, label_ds, lookup_table):
    doc_ds = doc_ds.map(extract_sents, 4)
    doc_ds = doc_ds.map(extract_words, 4)
    doc_ds = doc_ds.map(extract_chars, 4)
    doc_ds = doc_ds.map(lookup_table.lookup, 4)
    label_ds = label_ds.map(one_hot_multi_label, 4)

    return doc_ds, label_ds


if __name__ == "__main__":
    tf.set_random_seed(0)
    train_words_file = None
    train_docs_file = None
    train_labels_file = None

    padded_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None]))
    padding_values = (np.int64(0), np.int32(0))
    lookup_table = tf.contrib.lookup.index_table_from_file(
        train_words_file, num_oov_buckets=1
    )

    train_doc_ds = tf.data.TextLineDataset(train_docs_file)
    train_labels_ds = tf.data.TextLineDataset(train_labels_file)

    dds, lds = preprocess_dataset(train_doc_ds, train_labels_ds, lookup_table)
    train_dataset = tf.data.Dataset.zip((dds, lds))
    train_dataset = train_dataset.shuffle(10000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.padded_batch(32, padded_shapes, padding_values)
    train_dataset = train_dataset.prefetch(10)

    train_iter = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes
    )
    train_dataset_init_op = train_iter.make_initializer(
        train_dataset, name="train_dataset_init_op"
    )

    input_tensor, labels_tensor = train_iter.get_next()
