import pytest

from parao.core import Param, ParaO, Prop
from parao.steno import (
    InconsistentValues,
    MultipleSameValues,
    NoSuchParameter,
    Steno,
)


class Layer1(ParaO):
    foo = Param[int](0)
    bar = Param[int](0)


class Layer2(ParaO):
    left = Param[Layer1]()
    right = Param[Layer1]()


class Layer3(Steno, ParaO):
    inner = Param[Layer2]()


class Chain(Steno, ParaO):
    depth = Param[int]()
    aux = Param[int]()

    @Prop
    def inner(self) -> "Chain | None":
        if self.depth > 0:
            return Chain(depth=self.depth - 1)


def test_different():
    inst = Layer3({"foo": 1, ("right", "foo"): 2})
    assert inst["left", "foo"] == 1

    assert inst.right is inst.inner.right
    with pytest.raises(InconsistentValues):
        inst.foo
    with pytest.raises(NoSuchParameter):
        inst.wrong
    with pytest.raises(AttributeError):
        inst._ignored


def test_same():
    with pytest.warns(MultipleSameValues):
        assert Layer3({"foo": 3, ("right", "bar"): 4}).foo == 3


def test_identical():
    assert Layer3({"foo": 3}).foo == 3
    assert Layer3({"foo": 3}).foo == 3


def test_deep():
    ch = Chain({"depth": 3, (Chain, Chain.aux): 1})
    with pytest.warns(MultipleSameValues):
        assert ch["inner", "inner", "aux"] == 1
