import datetime
import json
import pickle as pkl
from pathlib import Path

from ..constants import padding_token, split_doc_token
from ..utils.utils import get_new_dir, is_prop

path_to_yelp = "/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/"


class HP:
    @staticmethod
    def from_dict(dic):
        h = HP()
        for k, v in dic.items():
            if k not in {"dir"} and not is_prop(h, k):
                setattr(h, k, v)
        h.set_dir(h._path)
        return h

    @staticmethod
    def json_serial(obj):
        try:
            return json.dumps(obj)
        except TypeError:
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            if "numpy" in str(type(obj)):
                return obj.tolist()
            raise TypeError("(HP:json_serial) Type %s not serializable" % type(obj))

    @staticmethod
    def load(path_to_HP, name="", file_type="json"):
        """Static method: returns a HP object, located at path_to_HP.
        If path_to_HP is a directory, will return the first .pkl file

        Args:
            path_to_HP ([type]): [description]

        Returns:
            [type]: [description]
        """
        path = Path(path_to_HP).resolve()
        to = file_type if file_type in ["pickle", "json"] else "pickle"
        if to == "pickle":
            ext = ".pkl"
        else:
            ext = ".json"

        if path.is_dir():
            files = [
                f for f in path.iterdir() if ext in str(f) and "%s_hp" % name in str(f)
            ]
            if not files:
                raise FileNotFoundError("No %s file in provided path" % ext)
            path = files[0]
        if to == "pickle":
            with path.open("rb") as f:
                hp = pkl.load(f)
        else:
            with path.open("r") as f:
                jhp = json.load(f)
            hp = HP.from_dict(jhp)

        hp.restored = True
        return hp

    def __init__(
        self,
        base_dir=None,
        batch_size=64,
        cell_size=5,
        chan=True,
        decay_rate=0.99,
        decay_steps=10,
        default_name="trainer",
        dense_activation="selu",
        dense_layers=[128, 128],
        doc_cell_size=None,
        doc_rnn_layers=None,
        dropout=0.6,
        dtype=32,
        embedding_dim=100,
        embedding_file="",
        epochs=20,
        experiments_dir="./experiments",
        fast_text_model_file=None,
        fast_text=False,
        global_step=0,
        learning_rate=5e-3,
        max_grad_norm=5.0,
        max_train_size=-1,
        max_val_size=-1,
        max_words=1e5,
        model_type="HAN",
        multilabel=False,
        num_classes=5,
        num_threads=4,
        pad_word=padding_token,
        ref_labels_delimiter=": ",
        ref_labels="",
        restored=False,
        restrict=-1,
        retrained=False,
        rnn_layers=6,
        save_steps=1000,
        sent_cell_size=None,
        sent_rnn_layers=None,
        split_doc_token=split_doc_token,
        stop_learning_count=1,  # strict, nonzeros < stop_learning_count
        stop_learning_epoch=2,  # stop after first validation in epoch + 1 (start 0)
        stop_learning=True,
        summary_secs=1000,
        train_docs_file=path_to_yelp + "sample_0001_train_07/documents.txt",
        train_labels_file=path_to_yelp + "sample_0001_train_07/labels.txt",
        train_words_file=path_to_yelp + "sample_0001_train_07/words.txt",
        trainable_embedding_matrix=True,
        use_bnlstm=True,
        val_batch_size=1000,
        val_docs_file=path_to_yelp + "sample_0001_val_01/documents.txt",
        val_every_steps=100,
        val_labels_file=path_to_yelp + "sample_0001_val_01/labels.txt",
        version="v2",
    ):

        now = datetime.datetime.now()
        self._str_no_k = ["dir"]
        self._path = ""
        self.created_at = int(now.timestamp())
        self.base_dir = base_dir or str(now)[:10]
        self.experiments_dir = Path(experiments_dir).resolve()

        self.batch_size = batch_size
        self.cell_size = cell_size
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.default_name = default_name
        self.dense_activation = dense_activation
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.dtype = dtype
        self.embedding_dim = embedding_dim
        self.embedding_file = embedding_file
        self.epochs = epochs
        self.fast_text = fast_text
        self.fast_text_model_file = fast_text_model_file
        self.global_step = global_step
        self.id = None
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.max_words = max_words
        self.model_type = model_type
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.num_threads = num_threads
        self.pad_word = pad_word
        self.path_initialized = False
        self.restored = restored
        self.restrict = restrict
        self.retrained = retrained
        self.rnn_layers = rnn_layers
        self.save_steps = save_steps
        self.split_doc_token = split_doc_token
        self.summary_secs = summary_secs
        self.train_docs_file = train_docs_file
        self.train_labels_file = train_labels_file
        self.train_words_file = train_words_file
        self.trainable_embedding_matrix = trainable_embedding_matrix
        self.use_bnlstm = use_bnlstm
        self.val_batch_size = val_batch_size
        self.val_docs_file = val_docs_file
        self.val_every_steps = val_every_steps
        self.val_labels_file = val_labels_file
        self.version = version
        self.vocab_size = None
        self.max_train_size = None
        self.max_val_size = max_val_size
        self.ref_labels = ref_labels
        self.ref_labels_delimiter = ref_labels_delimiter
        self.stop_learning = stop_learning
        self.stop_learning_count = stop_learning_count
        self.stop_learning_epoch = stop_learning_epoch
        self.chan = chan
        self.doc_cell_size = doc_cell_size
        self.sent_cell_size = sent_cell_size
        self.doc_rnn_layers = doc_rnn_layers
        self.sent_rnn_layers = sent_rnn_layers

        assert not (bool(self.chan) and bool(self.fast_text))

    def __str__(self):
        """Returns a string representation of the Hyperparameter:
        no methods, no functions are included. The 'dir' attribute
        is also ignored as it would create a directory if not already
        existing

        Returns:
            str: string representation of the hyperparameter
        """
        return "\n".join(
            "{:25s} : {:10s}".format(k, str(getattr(self, k)))
            for k in sorted(dir(self))
            if "__" not in k
            and k not in self._str_no_k
            and not callable(getattr(self, k))
            and not is_prop(self, k)
        )

    def __repr__(self):
        """Class name and id are enough to represent a hyper parameter

        Returns:
            str: <<classname> self.id>
        """
        return "<%s %s>" % (self.__class__, self.id)

    def safe_dict(self):
        d = {
            k: getattr(self, k)
            for k in sorted(dir(self))
            if k not in self._str_no_k
            and not is_prop(self, k)
            and not callable(getattr(self, k))
            and "__" not in k
        }
        return d

    def dump(self, name="", to="json"):
        """Dumps the Hyperparameter as a pickle file in hp.dir as:
        <name> + _hp.pkl

            name (str, optional): Defaults to ''. details to add to the
            default filename
        """
        to = to if to in ["pickle", "json", "all"] else "pickle"
        if to in ["pickle", "all"]:
            ext = ".pkl"
            file_name = "%s_hp%s" % (name, ext)
            location = self.dir / file_name
            with location.open("wb") as f:
                pkl.dump(self, f)
            if to == "all":
                to = "json"
        if to == "json":
            ext = ".json"
            file_name = "%s_hp%s" % (name, ext)
            location = self.dir / file_name
            with location.open("w") as f:
                json.dump(self.safe_dict(), f, default=self.json_serial)

    def set_dir(self, path):
        """Set the HP's directory: hp.dir

        Args:
            path (pathlib Path or str): new path for the HP's dir,
            pathlib compatible
        """
        self._path = Path(path).resolve()
        if not self._path.exists():
            self._path.mkdir()
        self.path_initialized = True

    def assign_new_dir(self):
        self.path_initialized = False
        self._path = self.dir

    @property
    def dir(self):
        """Returns a pathlib object whith the parameter's directory.
        If it's the first time the dir is accessed, it is created.
        Cannonical path is:
        currentworkingdirectory/models/<self.base_dir>_<new_index>

        Returns:
            pathlib Path: The HP's directory
        """

        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True)

        if not self.path_initialized:
            path = get_new_dir(self.base_dir, self.default_name)
            self.id = "{} | {}".format(self.version, path.name)
            path.mkdir()

            self._path = path
            self.path_initialized = True

        if not self._path.exists():
            self._path.mkdir(parents=True)

        return self._path

    def set_vocab_size(self, size):
        self.vocab_size = size
