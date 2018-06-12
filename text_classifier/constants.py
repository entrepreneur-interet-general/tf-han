split_doc_token = "|&|"
padding_token = "<PAD>"
randomizable_params = {
    "cell_size": [1, 5, 1, "range", "uniform"],
    "rnn_layers": [1, 2, 1, "range", "uniform"],
    "batch_size": [32, 64, 32, "range", "uniform"],
    "dropout": [0.5, 0.9, 0.05, "range", "uniform"],
    "embedding_dim": [1, 10, 5, "range", "uniform"],
    "epochs": [1, 2, 1, "range", "uniform"],
    "learning_rate": [
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        "list",
        "uniform",
    ],
}
metrics = ["micro_f1", "macro_f1", "weighted_f1", "accuracy"]
