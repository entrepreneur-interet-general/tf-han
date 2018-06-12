import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import metrics, randomizable_params
from .hyperparameters import HP
from .trainers import DST
from .utils import EndOfExperiment, get_new_dir, normal_choice, uniform_choice


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

    def __init__(self, experiments_dir="./experiments", conf_path=None, hp=None):

        self.conf_path = conf_path
        self.conf = None
        self.experiments_dir = Path(experiments_dir)
        self.log = None
        self.trainer = None
        self.params = None
        self.tee = None
        self.summary = {
            "params": {p: [] for p in randomizable_params},
            "other": {},
            "metrics": {m: [] for m in metrics},
        }
        self.current_run = 0

        if self.conf_path:
            with open(Path(conf_path).resolve(), "r") as f:
                self.conf = json.load(f)

        if self.conf:
            self.trainer_type = self.conf["trainer_type"]
            if self.conf["exp_id"]:
                self.exp_id = self.conf["exp_id"]
            else:
                self.exp_id = str(datetime.datetime.now())[:10]
        else:
            self.trainer_type = "DST"
            self.exp_id = str(datetime.datetime.now())[:10]

        if not self.experiments_dir.exists():
            print("Creating %s" % str(self.experiments_dir))
            self.experiments_dir.mkdir(parents=True)

        self.dir = get_new_dir(self.experiments_dir, self.exp_id)

    def randomize(self, conf=None, verbose=0):
        conf = conf or self.conf
        if conf:
            self.params = conf["randomizable_params"]
        else:
            self.params = randomizable_params

        for param in self.params:
            distribution = self.params[param]

            if distribution[-2] == "range":
                values = np.arange(*distribution[:3])
            else:
                values = np.array(distribution[:-2])

            if distribution[-1] == "normal":
                value = normal_choice(values)
            else:
                value = uniform_choice(values)
            setattr(self.trainer.hp, param, value.tolist())
            self.summary["params"][param].append(value)
            if verbose > 0:
                print("{:20}: {:10}".format(param, value))

    def summarize(self):
        metrics = pd.DataFrame(self.summary["metrics"])
        params = pd.DataFrame(self.summary["params"])
        other = "\n".join(
            "{:20}: {}".format(k, v) for k, v in self.summary["other"].items()
        )
        summary = "{}\n\n{}".format(
            other, pd.concat([metrics, params], axis=1).to_string()
        )
        with open(self.dir / "summary.txt", "a") as f:
            f.write(summary)
        metrics.to_csv(self.dir / "metrics.csv")
        params.to_csv(self.dir / "params.csv")

    def setup(self, log=True):
        if self.trainer_type == "DST":
            hp = HP(base_dir=self.dir)
            self.trainer = DST(hp=hp)
        else:
            raise ValueError("Unknown Trainer")

        self.log = log
        if log:
            self.tee = Tee(str(self.trainer.hp.dir / "log.txt"))
            sys.stdout = self.tee

    def reset(self):
        self.setup()
        self.randomize()

    def delete(self):
        shutil.rmtree(self.dir)

    def update_metrics(self, metrics):
        acc, mic, mac, wei = metrics
        self.summary["metrics"]["micro_f1"].append(mic)
        self.summary["metrics"]["macro_f1"].append(mac)
        self.summary["metrics"]["weighted_f1"].append(wei)
        self.summary["metrics"]["accuracy"].append(acc)

    def run(self, n_runs=np.iinfo(int).max, randomize=True, log=True, verbose=0):
        print("\n= = > Run", self.current_run)

        self.setup(log)

        if randomize:
            self.randomize(verbose=verbose)

        while self.current_run < n_runs:
            if self.current_run > 0:
                print("\n= = > Run", self.current_run)
            try:
                metrics = self.trainer.train()
                self.update_metrics(metrics)
            except EndOfExperiment:
                print("\nStopping experiment. Delete?")
                answer = input("y/n")
                if "y" in answer:
                    self.delete()
                    return
                else:
                    self.summary["other"]["interrupting"] = "Keyboard interrupted"
                    self.summarize()
                break

            self.current_run += 1
            if self.current_run < n_runs:
                self.reset()
            # End of run
        # End of all runs
        self.summary["other"]["interrupting"] = "Done: all runs performed."
        self.summarize()
