import json
import sys
from importlib import reload
from pathlib import Path
from random import normalvariate
from types import ModuleType

import matplotlib
import numpy as np
from sklearn.metrics import f1_score

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _reload(module, top_level_name):
    reload(module)
    # print("Reloading ", module)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType:
            if top_level_name in attribute.__name__:
                _reload(attribute, top_level_name)


def rreload(module):
    """Recursively reload modules."""
    name = module.__name__
    _reload(module, name)
    _reload(module, name)


def is_prop(obj, attr):
    return isinstance(getattr(type(obj), attr, None), property)


def normal_choice(lst, mean=None, stddev=None):
    """pick a value in a list, according to a normal distribution
    over the indexes (not the values) in this list: mean is the middle of the list
    and std is len(lst) / 5
    
    Args:
        lst (list or 1D array): values to choose value from
        mean (float, optional): Defaults to None. forced mean for the normal
            distribution. Should be set according to lst's length not values
        stddev (float, optional): Defaults to None. Same as mean, but for the std
    
    Returns:
        np.scalar: 1 value, float or int according to lst
    """
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 5

    while True:
        index = int(normalvariate(mean, stddev) + 0)
        if 0 <= index < len(lst):
            return lst[index]
    # To plot the distribution:
    # for p in randomizable_params:
    #     param = list(randomizable_params[p][0])
    #     vals = [normal_choice(param) for _ in range(100000)]
    #     plt.hist(vals, bins=param)
    #     plt.title(p)
    #     plt.show()


def uniform_choice(lst):
    """Pick a value in lst, uniformly
    
    Args:
        lst (list or 1D array): list of values from which to choose
    
    Returns:
        np.scalar: 1 value, float or int according to lst
    """
    return np.random.choice(lst)


def get_new_dir(path, default_name):
    """returns a Path to a  new directory with name `default_name` at location `path`
    If `default_name` is already a directory in `path` then _i is appended
    to `default_name`, `i` being the number of pre-existing directories
    
    Args:
        path (str or pathlib.Path): location where the directory should be created
        default_name (str): name of the new directory
    
    Returns:
        pathlib.Path: new directory to create
    """
    assert isinstance(default_name, str)

    path = Path(path)
    assert path.exists()

    paths = [
        p.resolve() for p in path.iterdir() if p.is_dir() and default_name in str(p)
    ]
    if paths:
        _id = (
            max(
                [0]
                + [
                    int(str(p.name).split("_")[-1] if "_" in str(p.name) else "0")
                    for p in paths
                ]
            )
            + 1
        )
        new_name = "{}_{}".format(default_name, _id)
    else:
        new_name = default_name

    path /= new_name

    return path.resolve()


class EndOfExperiment(Exception):
    """Exception raised when the experiment should be stopped
    """

    pass


class Tee:
    """Class used to both write to stdout and to a file
    to log prints
    """

    def __init__(self, path):
        """Creating the Tee with a path to the file
        
        Args:
            path (str or pathlib.Path): path to the file to write to
        """
        self.file = open(path, "a")
        self.stdout = sys.stdout

    def __del__(self):
        """When Tee is deleted, close the file and reset stdout
        """
        self.file.close()
        sys.stdout = self.stdout

    def write(self, *args, **kwargs):
        """Write to both the stdout and the file
        """
        self.file.write(*args, **kwargs)
        self.stdout.write(*args, **kwargs)
        self.flush()

    def flush(self):
        """flush both outs
        """
        self.file.flush()
        self.stdout.flush()

    def reset_stdout(self):
        """Reset stdout to default
        """
        sys.stdout = self.stdout


def thread_eval(ys, preds, directory, step, ref_labels, ref_labels_delimiter):
    step_string = "_{}".format(step if step is not None else "final")
    directory = Path(directory) / "step-{}".format(step)
    if not directory.exists():
        directory.mkdir(parents=True)

    thresholds = np.arange(0.1, 0.9, 0.05)
    averages = [None, "micro", "macro", "weighted"]

    metrics = {str(av): [] for av in averages}

    for av in averages:
        for threshold in thresholds:
            metrics[str(av)].append(
                np.array(f1_score(ys, preds > threshold, average=av)).tolist()
            )
    with open(directory / "logits_metrics{}.npy".format(step_string), "w") as f:
        json.dump(metrics, f)

    fig = plt.figure()
    st = fig.suptitle(
        "F1 scores - {}".format(
            "step {}".format(step) if step is not None else "final"
        ),
        fontsize="x-large",
    )
    for i, av in enumerate(averages[1:]):
        ax = fig.add_subplot("22{}".format(i + 1))
        ax.plot(thresholds, metrics[av])
        ax.set_title(av)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    fig.savefig(str(directory / "logits_metrics{}.png".format(step_string)), dpi=360)

    labels = {}
    with open(ref_labels, "r") as f:
        for l in f:
            l = l.strip()
            v, k = l.split(ref_labels_delimiter)
            labels[int(k)] = v
    x = np.arange(len(metrics["None"][0]))
    for met, threshold in zip(metrics["None"], thresholds):
        indexes = np.argsort(met)
        mets = np.sort(met)
        plt.figure(figsize=(12, 8))
        plt.suptitle("F1-score per class at threshold {:3f}".format(threshold))
        plt.bar(x, mets)
        plt.xticks(x, [labels[k] for k in indexes], rotation="vertical")
        plt.tight_layout()
        plt.savefig(
            str(
                directory
                / "logits_metrics_perclass_{:.3f}{}.png".format(threshold, step_string)
            ),
            dpi=360,
        )
        plt.close()
