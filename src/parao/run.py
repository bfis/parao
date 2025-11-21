from abc import ABC, abstractmethod
from concurrent.futures import Executor, wait
from typing import TYPE_CHECKING, Callable, Iterable, Self, overload

from .action import BaseRecursiveAction, RecursiveAct
from .cast import Opaque
from .core import ParaO
from .misc import ContextValue


class PseudoOutput(Opaque):
    """Special type that must only be handled directly lest it cause an error."""


class BaseOutput[R, A: RunAct](ABC):
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
    def __call__[R, A, I](self, act: "RunAct[R, A, I]") -> BaseOutput[R, A]: ...


class _Runnable(ParaO):
    output: _Template


class RunAct[R, A: RunAction, I: _Runnable](RecursiveAct[R, A, I]):
    __call__: Callable[[], R]

    def _func(
        self, sub: Iterable[Self], runner: "Runner[R] | None" = None, **kwargs
    ) -> R:
        if runner is None:
            runner = Runner.current()
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
    output: BaseOutput[R, Self]

    def __post_init__(self):
        super().__post_init__()
        output = self.instance.output(self)
        object.__setattr__(self, "output", output)  # side-step "frozen"

    @property
    def done(self):
        return self.output.exists

    @property
    def _key(self):
        return self.instance, self.action


class RunAction[R](BaseRecursiveAction[R, []]):
    if TYPE_CHECKING:

        @overload
        def __get__[I: ParaO](
            self, inst: I, owner: type | None = None
        ) -> RunAct[R, Self, I]: ...
        @overload
        def __get__(self, inst: None | RunAct, owner: type | None = None) -> Self: ...

    func: Callable[[ParaO], R]
    _act = RunAct


RunAction._peer_base = RunAction


class Runner[R](ABC):
    current = ContextValue["Runner | None"]("currentRunner", default=None)

    @abstractmethod
    def __call__(self, act: RunAct, sub: Iterable[RunAct], sub_kwargs: dict) -> R: ...


class ConcurrentRunner(Runner):
    def __init__(self, executor: Executor):
        self.executor = executor
        super().__init__()

    def __call__(self, act: RunAct, sub: Iterable[RunAct], sub_kwargs: dict):
        if act.done:
            return act.output.load()
        else:
            wait(self.executor.submit(s, **sub_kwargs) for s in sub if not s.done)
            return self.executor.submit(act._make).result()
