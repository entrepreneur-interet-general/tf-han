import numbers
from pathlib import Path

import numpy as np
import pytest

from ...utils.utils import Tee, get_new_dir, is_prop, normal_choice, uniform_choice


@pytest.fixture
def get_obj():
    class Obj(object):
        def __init__(self):
            self.test = 1
            self._hello = 2

        @property
        def hello(self):
            return self._hello

    yield Obj()


@pytest.fixture
def values():
    return list(range(30))


@pytest.mark.parametrize(
    "attr,expected",
    [("test", False), ("hello", True), ("_hello", False), ("__str__", False)],
)
def test_is_prop(get_obj, attr, expected):
    assert is_prop(get_obj, attr) == expected


def test_normal_choice_on_list(values):
    assert isinstance(normal_choice(values), numbers.Number)


def test_normal_choice_on_array(values):
    values = np.array(values)
    assert isinstance(normal_choice(values), numbers.Number)


def test_uniform_choice_on_list(values):
    assert isinstance(uniform_choice(values), numbers.Number)


def test_uniform_choice_on_array(values):
    values = np.array(values)
    assert isinstance(uniform_choice(values), numbers.Number)


@pytest.mark.parametrize("name", [None, 1, True, "Hello"])
def test_get_new_dir_name_is_string(name):
    if not isinstance(name, str):
        with pytest.raises(AssertionError):
            get_new_dir(".", name)
    else:
        assert isinstance(get_new_dir(".", name), Path)


def test_get_new_dir_from_string():
    assert isinstance(get_new_dir(".", "name"), Path)


def test_get_new_dir_from_path():
    assert isinstance(get_new_dir(Path("."), "name"), Path)


def test_get_new_dir_no_pre_existing(tmpdir):
    tdir = str(tmpdir)
    assert str(get_new_dir(Path(tdir), "hello")) == str(tmpdir.join("hello"))


def test_get_new_dir_with_pre_existing(tmpdir):
    tdir = str(tmpdir)
    newdir = tmpdir.mkdir("newdir")
    assert str(get_new_dir(Path(tdir), "newdir")) == str(newdir) + "_1"


def test_get_new_dir_increment(tmpdir):
    tdir = str(tmpdir)
    newdir = tmpdir.mkdir("newdir_1")
    assert str(get_new_dir(Path(tdir), "newdir")) == str(newdir)[:-1] + "2"
