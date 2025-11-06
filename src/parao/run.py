from concurrent.futures import Executor, wait
from typing import Callable, Iterable, Self, Type

from .action import BaseRecursiveAction, RecursiveAct
from .cast import Opaque
from .core import ParaO
from .misc import ContextValue


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

    def _func(
        self, sub: Iterable[Self], runner: "Runner[T] | None" = None, **kwargs
    ) -> T:
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
    output: BaseOutput

    def __post_init__(self):
        super().__post_init__()
        output = self.action.output or self.instance.output
        if not (isinstance(output, type) and issubclass(output, BaseOutput)):
            raise TypeError(f"{output=} must by a BaseOutput subclass")
        object.__setattr__(self, "output", output(self))  # side-step "frozen"

    @property
    def done(self):
        return self.output.exists


class RunAction[R](BaseRecursiveAction[R, []]):
    _act: Type[RunAct[R]] = RunAct
    func: Callable[[ParaO], R]
    __get__: Callable[..., RunAct[R]]
    output: Type[BaseOutput] | None = None


class Runner:
    current = ContextValue["Runner | None"]("currentRunner", default=None)

    def __call__[T](self, act: RunAct[T], sub: Iterable[RunAct], sub_kwargs: dict):
        raise NotImplementedError  # pragma: no cover


class ConcurrentRunner(Runner):
    def __init__(self, executor: Executor):
        self.executor = executor
        super().__init__()

    def __call__[T](self, act: RunAct[T], sub: Iterable[RunAct], sub_kwargs: dict):
        if act.done:
            return act.output.load()
        else:
            wait(self.executor.submit(s, **sub_kwargs) for s in sub if not s.done)
            return self.executor.submit(act._make).result()
