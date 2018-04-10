import os
import pathlib
import datetime
import pickle as pkl
import json


class HP(object):

    @staticmethod
    def from_dict(dic):
        h = HP()
        for k, v in dic.items():
            if k not in {'dir'}:
                h.__setattr__(k, v)
        h.set_dir(h._path)
        return h

    @staticmethod
    def json_serial(obj):
        try:
            return json.dumps(obj)
        except TypeError:
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, pathlib.Path):
                return str(obj)
            raise TypeError("Type %s not serializable" % type(obj))

    @staticmethod
    def load(path_to_HP, file_type="pickle"):
        """Static method: returns a HP object, located at path_to_HP.
        If path_to_HP is a directory, will return the first .pkl file

        Args:
            path_to_HP ([type]): [description]

        Returns:
            [type]: [description]
        """
        path = pathlib.Path(path_to_HP)
        to = file_type if file_type in["pickle", "json"] else "pickle"
        if to == "pickle":
            ext = '.pkl'
        else:
            ext = '.json'

        if path.is_dir():
            files = [
                f for f in path.iterdir()
                if ext in str(f)
            ]
            if not files:
                raise FileNotFoundError('No %s file in provided path' % ext)
            path = files[0]
        if to == "pickle":
            with path.open('rb') as f:
                hp = pkl.load(f)
        else:
            with path.open('r') as f:
                jhp = json.load(f)
            hp = HP.from_dict(jhp)
        return hp

    def __init__(
            self,
            base_dir_name=None,
            batch_size=128,
            save_steps=1000,
            summary_secs=1000,
            version='v1',
    ):
        self.__str_no_k = set(['dir'])
        self.__str_no_v = set(['method', 'function'])
        self._path = ''
        self.version = version
        self.save_steps = save_steps
        self.summary_secs = summary_secs
        self.batch_size = batch_size
        self._max_sent_len = None
        self._max_doc_len = None
        self.path_initialized = False
        self.now = datetime.datetime.now()
        self.id = None

        self.base_dir_name = base_dir_name or str(self.now)[:10]

    def __str__(self):
        """Returns a string representation of the Hyperparameter:
        no methods, no functions are included. The 'dir' attribute
        is also ignored as it would create a directory if not already
        existing

        Returns:
            str: string representation of the hyperparameter
        """
        return '\n'.join(
            '{:25s} : {:10s}'.format(k, str(self.__getattribute__(k)))
            for k in sorted(dir(self))
            if '__' not in k and
            k not in self.__str_no_k and
            not any(v in str(self.__getattribute__(k))
                    for v in self.__str_no_v)
        )

    def __repr__(self):
        """Class name and id are enough to represent a hyper parameter

        Returns:
            str: <<classname> self.id>
        """
        return '<%s %s>' % (self.__class__, self.id)

    def safe_dict(self):
        d = {
            k: self.__getattribute__(k)
            for k in dir(self)
            if '__' not in k and
            k not in self.__str_no_k and
            not any(v in str(self.__getattribute__(k))
                    for v in self.__str_no_v)
        }
        return d

    def dump(self, name='', to="json"):
        """Dumps the Hyperparameter as a pickle file in hp.dir as:
        <name> + _hp.pkl

            name (str, optional): Defaults to ''. details to add to the
            default filename
        """
        to = to if to in["pickle", "json", "all"] else "pickle"
        if to in ["pickle", "all"]:
            ext = '.pkl'
            file_name = '%s_hp%s' % (name, ext)
            location = self.dir / file_name
            with location.open('wb') as f:
                pkl.dump(self, f)
            if to == "all":
                to = "json"
        if to == "json":
            ext = '.json'
            file_name = '%s_hp%s' % (name, ext)
            location = self.dir / file_name
            with location.open('w') as f:
                json.dump(self.safe_dict(), f, default=self.json_serial)

    def set_dir(self, path):
        """Set the HP's directory: hp.dir

        Args:
            path (pathlib Path or str): new path for the HP's dir,
            pathlib compatible
        """
        self._path = pathlib.Path(path)
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
        current/working/directory/checkpoints/
            <self.version>/<self.base_dir_name>_<new_index>

        Returns:
            pathlib Path: The HP's directory
        """
        cwd = os.getcwd()
        cpath = pathlib.Path(cwd)
        ckpt_dir = cpath / 'checkpoints'
        path = ckpt_dir / self.version
        if not path.exists():
            path.mkdir(parents=True)

        if not self.path_initialized:
            paths = [
                p.resolve() for p in path.iterdir()
                if p.is_dir() and
                self.base_dir_name in str(p)
            ]
            if paths:
                _id = max(
                    [0] + [
                        int(str(p).split("_")[-1] if '_' in str(p) else "0")
                        for p in paths]
                ) + 1
                new_name = self.base_dir_name + '_%d' % _id
                self.id = self.version + ' | ' + new_name
            else:
                new_name = self.base_dir_name
                self.id = self.version + ' | ' + new_name

            path /= new_name
            path.mkdir()

            self._path = path
            self.path_initialized = True

        if not self._path.exists():
            self._path.mkdir(parents=True)

        return self._path

    @property
    def max_sent_len(self):
        if self._max_sent_len:
            return self.max_sent_len
        raise NotImplementedError('Set max_sent_len first')

    def set_max_sent_len(self, length):
        self._max_sent_len = length

    @property
    def max_doc_len(self):
        if self._max_doc_len:
            return self.max_doc_len
        raise NotImplementedError('Set max_doc_len first')

    def set_max_doc_len(self, length):
        self._max_doc_len = length
