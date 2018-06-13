import pytest
from ..utils.utils import is_prop, normal_choice, uniform_choice, get_new_dir, Tee


@pytest.fixture
def get_obj():
    class Obj(object):
        def __init__(self):
            self.test = 1
            self._hello = 2

        @property
        def hello(self):
            return self._hello

    return Obj()


@pytest.mark.parametrize(
    "attr,expected",
    [("test", False), ("hello", True), ("_hello", False), ("__str__", False)],
)
def test_is_prop(get_obj, attr, expected):
    assert is_prop(get_obj, attr) == expected
