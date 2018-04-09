import os
import pathlib
import datetime
import pickle as pkl


class HP(object):

    @staticmethod
    def load(path_to_HP):
        """[summary]
        
        Args:
            path_to_HP ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        path = pathlib.Path(path_to_HP)
        with path.open('rb') as f:
            hp = pkl.load(f)
        return hp

    def __init__(
            self,
            version='v1',
            base_dir_name=None):
        self._path = ''
        self.version = version
        self.path_initialized = False
        self.now = datetime.datetime.now()

        self.base_dir_name = base_dir_name or "%d-%d-%d" % (
            self.now.year, self.now.month, self.now.day
        )

    def save(self, name=''):
        """Dumps the Hyperparameter as a pickle file in hp.dir as:
        <name> + _hp.pkl

            name (str, optional): Defaults to ''. details to add to the
            default filename
        """
        file_name = '%s_hp.pkl' % name
        location = self.dir / file_name
        with location.open('wb') as f:
            pkl.dump(self, f)

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
                        int(str(p).split("_")[-1])
                        for p in paths if '_' in str(p)]
                ) + 1
                path /= self.base_dir_name + '_' + str(_id)
            else:
                path /= self.base_dir_name
            path.mkdir()
            self._path = path
            self.path_initialized = True

        if not self._path.exists():
            self._path.mkdir(parents=True)

        return self._path


class MHP(HP):
    pass


class THP(HP):
    pass
