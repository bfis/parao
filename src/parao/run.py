from functools import cached_property
from typing import Callable, Iterable, Self, Type

from .action import BaseRecursiveAction, RecursiveAct
from .cast import Opaque
from .core import ParaO


class PseudoOutput(Opaque):
    """Special type that must only be handled directly lest it cause an error."""


class BaseOutput[T]:
    __slots__ = ("act",)

    def __init__(self, act: "RunAct[T]"):
        self.act = act

    @property
    def exists(self) -> bool:
        raise NotImplementedError  # pragma: no cover

    def load(self) -> T:
        raise NotImplementedError  # pragma: no cover

    def dump(self, data: T) -> T:
        raise NotImplementedError  # pragma: no cover

    def remove(self, missing_ok: bool = False) -> None:
        raise NotImplementedError  # pragma: no cover


class RunAct[T](RecursiveAct["RunAction[T]"]):
    __call__: Callable[[], T]

    def _func(self, sub: Iterable[Self], depth, outer) -> T:
        if self.output.exists:
            return self.output.load()
        else:  # TODO: here we need to shim in Pool.map/asnyc
            for s in sub:
                if not s.done:
                    s(depth=depth + 1, outer=outer)

            return self.output.dump(self.action.func(self.instance))

    @cached_property
    def output(self) -> BaseOutput[T]:
        output = self.action.output or self.instance.output
        if not (isinstance(output, type) and issubclass(output, BaseOutput)):
            raise TypeError(f"{output=} must by a BaseOutput subclass")
        return output(self)

    @property
    def done(self):
        return self.output.exists


class RunAction[R](BaseRecursiveAction[R, []]):
    _act: Type[RunAct[R]] = RunAct
    func: Callable[[ParaO], R]
    __get__: Callable[..., RunAct[R]]
    output: Type[BaseOutput] | None = None
