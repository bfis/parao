from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, Future
from graphlib import TopologicalSorter
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Literal, Self, overload

from .action import RecursiveAct, _RecursiveAction
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
        self, sub: Iterable[Self], runner: "Runner[R] | None" = None, **kwargs
    ) -> R:
        if runner is None:
            runner = Runner.current()
        return runner(self, sub, kwargs)

    @overload
    def produce(self, output: Literal[False]) -> None: ...
    @overload
    def produce(self, output: bool = True) -> R: ...

    def produce(self, output: bool = True):
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


class Runner[R](ABC):
    __slots__ = ()
    current: ContextValue[Self]

    def __call__(self, act: _RunAct, sub: Iterable[_RunAct], aux: dict) -> R:
        if act.done:
            return act.output.load()
        else:
            for s in sub:
                if not s.done:
                    s(**aux)
            return act.produce()

    def scoped_produce(self, act: _RunAct, return_output: bool = True):
        with self.current(self):
            return act.produce(output=return_output)


Runner.current = ContextValue[Runner]("currentRunner", default=Runner())


class _TopoRunner[R](Runner[R]):
    __slots__ = ()
    _cls: type[TopologicalSorter] = TopologicalSorter

    def __call__(self, act, sub, aux):
        if act.done:
            assert "topo" not in aux
            return act.output.load()

        if root := "topo" not in aux:
            aux["topo"] = self._cls()
        aux["runner"] = self
        topo: TopologicalSorter[_RunAct] = aux["topo"]

        for t in (todo := [s for s in sub if not s.done]):
            t(**aux)

        topo.add(act, *todo)
        if root:
            return self.resolve(topo, act)

    @abstractmethod
    def resolve(self, topo: TopologicalSorter[_RunAct], root: _RunAct) -> R: ...


class ConcurrentRunner(_TopoRunner):
    def __init__(self, executor: Executor):
        self.executor = executor
        super().__init__()

    def resolve(self, topo, root):
        run = Runner().scoped_produce
        waiting: dict[Future, _RunAct] = {}
        queue = SimpleQueue[_RunAct]()
        topo.prepare()
        while topo.is_active():
            for act in topo.get_ready():
                future = self.executor.submit(run, act, act is root)
                future.add_done_callback(queue.put_nowait)
                waiting[future] = act
            wait = True
            while True:
                try:
                    future = queue.get(block=wait)
                except Empty:
                    break
                act = waiting.pop(future)
                result = future.result()
                topo.done(act)
                wait = False
        assert root is act
        return result
