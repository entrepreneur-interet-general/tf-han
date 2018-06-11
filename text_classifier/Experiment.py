import sys

from .hyperparameters import HP
from .trainers import DST


class Tee:
    def write(self, *args, **kwargs):
        self.file.write(*args, **kwargs)
        self.stdout.write(*args, **kwargs)
        self.flush()
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __init__(self, path):
        self.file = open(path, "a")
        self.stdout = sys.stdout

    def __del__(self):
        self.file.close()
        sys.stdout = self.stdout

    def reset_stdout(self):
        sys.stdout = self.stdout


class Experiment(object):
    @staticmethod
    def from_json(path):
        hp = HP.load(path, file_type="json")
        return Experiment(hp.experiments_dir, hp)

    def __init__(self, experiments_dir=".", hp=None):
        self.hp = hp or HP(experiments_dir=experiments_dir)
        self.log = None
        self.trainer = None
        self.tee = False

    def setup(self, log=True):
        self.log = log
        if log:
            self.tee = Tee(str(self.hp.dir / "log.txt"))
            sys.stdout = self.tee
        if self.hp.trainer_type == "DST":
            self.trainer = DST(self.hp.model_type, self.hp)
        else:
            raise ValueError("Unknown Trainer")

    def run(self):
        self.trainer.train()
