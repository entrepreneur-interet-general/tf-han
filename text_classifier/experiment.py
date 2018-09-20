import datetime
import shutil
import sys
from pathlib import Path
from gensim.models.wrappers import FastText

import numpy as np
import pandas as pd
import yaml
from munch import DefaultMunch

from scipy.special import expit
from sklearn.metrics import f1_score

from .constants import metrics
from .hyperparameters import HP
from .trainers import DST
from .trainers import FT_DST
from .trainers import CDST
from .utils import default_conf
from .utils.utils import (
    EndOfExperiment,
    get_new_dir,
    normal_choice,
    uniform_choice,
    Tee,
)


class Experiment(object):
    def __init__(self, conf_path=None, experiments_dir=None, hp=None):

        self.conf_path = conf_path
        self.conf = None
        self.log = None
        self.trainer = None
        self.tee = None
        self.current_run = 0

        self.fast_text_model = None

        if self.conf_path:
            self.set_conf(self.conf_path)
        else:
            self.conf = DefaultMunch(None).fromDict(default_conf)

        if experiments_dir:
            self.conf.experiments_dir = experiments_dir
        assert self.conf.experiments_dir is not None
        self.conf.experiments_dir = Path(self.conf.experiments_dir).resolve()

        if not self.conf.experiments_dir.exists():
            print("Creating %s" % str(self.conf.experiments_dir))
            self.conf.experiments_dir.mkdir(parents=True)

        if not self.conf.exp_id:
            self.conf.exp_id = str(datetime.datetime.now())[:10]

        self.dir = get_new_dir(self.conf.experiments_dir, self.conf.exp_id)

        self.summary = {
            "params": {p: [] for p in self.conf.randomizable_params},
            "other": {},
            "metrics": {m: [] for m in metrics},
        }

    def set_conf(self, path):
        with open(path, "r") as f:
            conf = yaml.safe_load(f)
        self.conf = DefaultMunch(None).fromDict(conf)

    def randomize(self, conf=None, verbose=0):
        conf = conf or self.conf
        params = conf.randomizable_params

        for p_name, p in params.items():
            if self.conf.trainer_type == "FT_DST" and p_name == "embedding_dim":
                self.summary["params"][p_name].append(300)
                continue
            if p.type == "range":
                values = np.arange(p.min, p.max, p.step)
            elif p.type == "list":
                values = np.array(p.vals)
            elif p.type == "fixed":
                value = np.array(p.value)
            else:
                raise ValueError("Unkonw type {} for {}".format(p.type, p_name))
            if p.type != "fixed":
                if p.distribution == "normal":
                    value = normal_choice(values)
                elif p.distribution == "uniform":
                    value = uniform_choice(values)
                elif p.distribution == "deterministic":
                    value = values[self.current_run % len(values)]

            setattr(self.trainer.hp, p_name, value.tolist())
            self.summary["params"][p_name].append(value)
            if verbose > 0:
                print("{:20}: {:10}".format(p_name, value))

    def dump_conf(self, path):
        stringified = []
        for attr, val in self.conf.items():
            if isinstance(val, Path):
                self.conf[attr] = str(val)
                stringified.append(attr)
        with open(path, "w") as f:
            yaml.safe_dump(self.conf, f, default_flow_style=False)
        for attr in stringified:
            self.conf[attr] = Path(self.conf[attr])

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
        self.dump_conf(self.dir / "conf.yaml")

    def setup(self, log=True):
        hp = HP(base_dir=self.dir)
        if self.conf.trainer_type == "DST":
            for attr, val in self.conf.hyperparameter.items():
                if val is not None:
                    setattr(hp, attr, val)
            self.trainer = DST(hp=hp)
        elif self.conf.trainer_type == "CDST":
            for attr, val in self.conf.hyperparameter.items():
                if val is not None:
                    setattr(hp, attr, val)
            self.trainer = CDST(hp=hp)
        elif self.conf.trainer_type == "FT_DST":
            for attr, val in self.conf.hyperparameter.items():
                if attr != "embedding_dim":
                    if val is not None:
                        setattr(hp, attr, val)
            if not self.fast_text_model:
                print("Setting fast_text_model...", end="")
                self.fast_text_model = FastText.load_fasttext_format(
                    hp.fast_text_model_file
                )
                print("Ok.")
            self.trainer = FT_DST(fast_text_model=self.fast_text_model, hp=hp)
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
        if metrics is None:
            metrics = None, None, None, None
        acc, mic, mac, wei = metrics
        self.summary["metrics"]["micro_f1"].append(mic)
        self.summary["metrics"]["macro_f1"].append(mac)
        self.summary["metrics"]["weighted_f1"].append(wei)
        self.summary["metrics"]["accuracy"].append(acc)

    def get_samples(self, samples, sample_size, is_val=False):
        preds, ys = None, None
        for _ in range(samples):
            x, y = self.trainer.get_input_pair(is_val, sample_size)
            pred = self.trainer.infer(x)
            if preds is None:
                preds, ys = pred, y
            else:
                preds = np.concatenate((preds, pred), axis=0)
                ys = np.concatenate((ys, y), axis=0)
        return expit(preds), ys

    def eval(self, thresholds, samples, sample_size, is_val=False):
        preds, ys = self.get_samples(samples, sample_size, is_val)
        averages = [None, "micro", "macro", "weighted"]

        metrics = {str(av): [] for av in averages}

        for av in averages:
            for threshold in thresholds:
                metrics[str(av)].append(f1_score(ys, preds > threshold, average=av))
        return metrics

    def run(self, n_runs=None, randomize=True, log=True, verbose=0):
        n_runs = n_runs or self.conf.n_runs
        if n_runs is None:
            n_runs = np.iinfo(int).max
        print("\n= = > Run", self.current_run)

        self.setup(log)

        if randomize:
            self.randomize(verbose=verbose)

        while self.current_run < n_runs:
            if self.current_run > 0:
                print("\n= = > Run", self.current_run)
            try:
                metrics = self.trainer.train()
                self.trainer.dump_logits()
                self.update_metrics(metrics)
                self.summarize()
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
