import datetime
import json
import os
import pickle as pkl
from pathlib import Path

from ..utils import is_prop


class HP(object):
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
            raise TypeError("(HP) Type %s not serializable" % type(obj))

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
        experiments_dir="./experiments",
        base_dir_name=None,
        batch_size=64,
        cell_size=5,
        decay_steps=10,
        decay_rate=0.99,
        dropout=0.6,
        embedding_file="",
        embedding_dim=100,
        epochs=20,
        global_step=0,
        learning_rate=5e-3,
        max_grad_norm=5.0,
        max_words=1e5,
        model_type='HAN',
        multilabel=False,
        num_classes=5,
        num_threads=4,
        pad_word="<PAD>",
        restored=False,
        retrained=False,
        rnn_layers=6,
        save_steps=1000,
        split_doc_token="|&|",
        summary_secs=1000,
        trainable_embedding_matrix=True,
        val_batch_size=1000,
        val_every=10,
        version="v2",
        val_docs_file="/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/sample_0001_val_01/documents.txt",
        val_labels_file="/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/sample_0001_val_01/labels.txt",
        train_docs_file="/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/sample_0001_train_07/documents.txt",
        train_labels_file="/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/sample_0001_train_07/labels.txt",
        train_words_file="/Users/victor/Documents/Tracfin/dev/han/data/yelp/tf-prepared/sample_0001_train_07/words.txt",
        trainer_type="DST"
    ):

        now = datetime.datetime.now()
        self._str_no_k = ["dir"]
        self._path = ""
        self.created_at = int(now.timestamp())
        self.base_dir_name = base_dir_name or str(now)[:10]
        self.experiments_dir = Path(experiments_dir).resolve()

        self.version = version
        self.save_steps = save_steps
        self.summary_secs = summary_secs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.val_every = val_every
        self.trainable_embedding_matrix = trainable_embedding_matrix
        self.cell_size = cell_size
        self.embedding_file = embedding_file
        self.max_words = max_words
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.multilabel = multilabel
        self.val_batch_size = val_batch_size
        self.max_grad_norm = max_grad_norm
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.global_step = global_step
        self.restored = restored
        self.retrained = retrained
        self.val_docs_file = val_docs_file
        self.val_labels_file = val_labels_file
        self.train_docs_file = train_docs_file
        self.train_labels_file = train_labels_file
        self.train_words_file = train_words_file
        self.num_threads = num_threads
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.trainer_type = trainer_type
        self.model_type = model_type
        self.pad_word = pad_word
        self.split_doc_token = split_doc_token

        self.vocab_size = None
        self.path_initialized = False
        self.id = None

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
        currentworkingdirectory/models/
            <self.version>/<self.base_dir_name>_<new_index>

        Returns:
            pathlib Path: The HP's directory
        """
        path = self.experiments_dir / self.version
        path /= self.base_dir_name
        if not path.exists():
            path.mkdir(parents=True)

        if not self.path_initialized:
            paths = [
                p.resolve()
                for p in path.iterdir()
                if p.is_dir() and self.base_dir_name in str(p)
            ]
            if paths:
                _id = (
                    max(
                        [0]
                        + [
                            int(str(p).split("_")[-1] if "_" in str(p) else "0")
                            for p in paths
                        ]
                    )
                    + 1
                )
                new_name = "{}_{}".format(self.base_dir_name, _id)
            else:
                new_name = self.base_dir_name

            self.id = "{} | {}".format(self.version, new_name)

            path /= new_name
            path.mkdir()

            self._path = path
            self.path_initialized = True

        if not self._path.exists():
            self._path.mkdir(parents=True)

        return self._path

    def set_vocab_size(self, size):
        self.vocab_size = size
