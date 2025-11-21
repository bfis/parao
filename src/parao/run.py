from abc import ABC, abstractmethod
from concurrent.futures import Executor, wait
from typing import TYPE_CHECKING, Callable, Iterable, Self, overload

from .action import _RecursiveAction, RecursiveAct
from .cast import Opaque
from .core import ParaO
from .misc import ContextValue


class PseudoOutput(Opaque):
    """Special type that must only be handled directly lest it cause an error."""


class _Output[R, A: _RunAct](ABC):
    __slots__ = ("act",)

    def __init__(self, act: A):
        self.act = act

    @property
    @abstractmethod
    def exists(self) -> bool: ...

    @abstractmethod
    def load(self) -> R: ...

    @abstractmethod
    def dump(self, data: R | PseudoOutput) -> R: ...

    @abstractmethod
    def remove(self, missing_ok: bool = False) -> None: ...


class _Template(ParaO):
    @abstractmethod
    def __call__[R, A, I](self, act: "_RunAct[R, A, I]") -> _Output[R, A]: ...


class _RunAct[R, A: _RunAction, I: ParaO](RecursiveAct[R, A, I]):
    __call__: Callable[[], R]

    def _func(
        self, sub: Iterable[Self], runner: "_Runner[R] | None" = None, **kwargs
    ) -> R:
        if runner is None:
            runner = _Runner.current()
        if runner is None:
            if self.done:
                return self.output.load()
            else:
                for s in sub:
                    if not s.done:
                        s(**kwargs)

                return self._make()
        else:
            return runner(self, sub, kwargs)

    def _make(self, output: bool = True):
        out = self.output.dump(self.action.func(self.instance))
        if output:
            return out

    __slots__ = ("output",)
    output: _Output[R, Self]

    def __post_init__(self):
        super().__post_init__()
        output = getattr(self.instance, self.action.output_template_attribute_name)
        object.__setattr__(self, "output", output(self))  # side-step "frozen"

    @property
    def done(self):
        return self.output.exists

    @property
    def _key(self):
        return self.instance, self.action


class _RunAction[R](_RecursiveAction[R, []]):
    if TYPE_CHECKING:

        @overload
        def __get__[I: ParaO](
            self, inst: I, owner: type | None = None
        ) -> _RunAct[R, Self, I]: ...
        @overload
        def __get__(self, inst: None | _RunAct, owner: type | None = None) -> Self: ...

    func: Callable[[ParaO], R]
    output_template_attribute_name: str
    _act = _RunAct


_RunAction._peer_base = _RunAction


class _Runner[R](ABC):
    current = ContextValue["_Runner | None"]("currentRunner", default=None)

    @abstractmethod
    def __call__(self, act: _RunAct, sub: Iterable[_RunAct], sub_kwargs: dict) -> R: ...


class ConcurrentRunner(_Runner):
    def __init__(self, executor: Executor):
        self.executor = executor
        super().__init__()

    def __call__(self, act: _RunAct, sub: Iterable[_RunAct], sub_kwargs: dict):
        if act.done:
            return act.output.load()
        else:
            wait(self.executor.submit(s, **sub_kwargs) for s in sub if not s.done)
            return self.executor.submit(act._make).result()
