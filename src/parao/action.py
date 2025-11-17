from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from inspect import Parameter, signature
from operator import attrgetter
from typing import Any, Callable, Concatenate, Iterable, Self, Type, overload

from .core import UNSET, AbstractDecoParam, ParaO, TypedAlias, Unset, Value, eager
from .misc import ContextValue

__all__ = ["SimpleAction", "ValueAction", "RecursiveAction"]


@lru_cache
def _method_1st_arg_annotation[T](
    func: Callable[Concatenate[Any, T, ...], Any],
) -> Type[T] | Unset:
    for i, param in enumerate(signature(func).parameters.values()):
        if i == 1:
            if param.kind in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_POSITIONAL,
            ):
                if param.annotation is Parameter.empty:
                    return UNSET
                else:
                    return param.annotation
            break
    return UNSET


@dataclass(slots=True, frozen=True)
class BaseAct[T, A: "BaseAction"](ABC):
    action: A
    instance: ParaO
    value: T
    position: int = 0

    def __post_init__(self):
        if self.trigger:
            Plan.add(self)

    @property
    def trigger(self):
        return self.value is not UNSET

    @property
    def name(self) -> str:
        return self.action._name(self.instance.__class__)

    @abstractmethod
    def __call__(self): ...


class BaseAction[T, R, **Ps](AbstractDecoParam[T, Callable[Concatenate[ParaO, Ps], R]]):
    significant = False
    _act: Type[BaseAct] = BaseAct

    TypedAlias.register(R, "return_type")

    def _type(self, cls, name):
        return self.type

    def _get(self, val, name, instance) -> BaseAct:
        pos = val.position if isinstance(val, Value) else 0
        val = super()._get(val, name, instance)
        return self._act(self, instance, val, pos)

    def _collect(self, expansion, instance):  # can't collect
        return False  # pragma: no cover

    @overload
    def __get__(self, inst: ParaO, owner: type | None = None) -> BaseAct[T, Self]: ...
    @overload
    def __get__(self, inst: None, owner: type | None = None) -> Self: ...

    del __get__  # don't overwrite the undelying __get__


# simple variant
class SimpleAct[R](BaseAct[bool, "SimpleAction[R]"]):
    __slots__ = ()

    @property
    def trigger(self):
        return super().trigger and self.value

    def __call__(self) -> R:
        return self.action.func(self.instance)


class SimpleAction[R](BaseAction[bool, R, []]):
    _act = SimpleAct
    func: Callable[[ParaO], R]
    __get__: Callable[..., SimpleAct[R]]
    type = bool


# value variant
class ValueAct[T, R](BaseAct[T, "ValueAction[T, R]"]):
    __slots__ = ()

    def __call__(self, override: T | Unset = UNSET) -> R:
        value = self.value if override is UNSET else override
        if value is UNSET:
            return self.action.func(self.instance)
        else:
            return self.action.func(self.instance, value)


class ValueAction[T, R](BaseAction[T, R, [T]]):
    def _type(self, cls, name):
        typ = self.type
        if typ is UNSET:
            typ = _method_1st_arg_annotation(self.func)
        return typ

    _act = ValueAct
    func: Callable[[ParaO, T], R]
    __get__: Callable[..., ValueAct[T, R]]
    type: Type[T]


# recursive variant
class RecursiveAct[A: "BaseRecursiveAction"](BaseAct[int | bool | None, A]):
    __slots__ = ()

    def _inner(self):
        name = self.name
        is_peer = self.action.__class__._is_peer
        for inner in self.instance.__inner__:
            if other := inner.__class__.__own_parameters__.get(name):
                if is_peer(other.__class__):
                    yield getattr(inner, name)

    def _func(self, sub: Iterable[Self], depth: int = 0, **kwargs):
        if not self.action.func(self.instance, depth):
            for s in sub:
                s(depth=depth + 1, **kwargs)

    def __call__(
        self, override: int | bool | None = None, *, _outer: int = None, **kwargs
    ):
        if override is None:
            val = self.value
            if val is UNSET:
                val = True if _outer is None else _outer
            elif self.trigger:  # pragma: no branch
                Plan.consume(self)
        else:
            val = override
        if val is False or val < 0:
            return

        return self._func(
            self._inner() if val else (),  # recusion elements, if (still) allowed
            _outer=val is True or val < 1 or val - 1,  # remaining recursion
            **kwargs,  # arbitrary other state, e.g. depth
        )


class BaseRecursiveAction[R, **Ps](BaseAction[int | bool | None, R, Ps]):
    _peer_base: type | None = None

    @classmethod
    def _is_peer(cls, other_cls: type):
        if base := cls._peer_base:
            return issubclass(other_cls, base)
        else:
            return other_cls is cls

    type = int | bool | None


class RecursiveAction(BaseRecursiveAction[bool, [int]]):
    func: Callable[[ParaO, int], bool]
    __get__: Callable[..., RecursiveAct]
    _act = RecursiveAct


class Plan(list[BaseAct]):
    current = ContextValue["Plan"]("currentPlan", default=None)
    _sorted: bool = False

    @classmethod
    def add(cls, act: BaseAct):
        if (curr := cls.current()) is not None:
            curr.append(act)
            curr._sorted = False

    @classmethod
    def consume(cls, act: BaseAct):
        if not (curr := cls.current()):
            return
        try:
            idx = curr.index(act)
        except ValueError:
            return
        name = act.name
        is_peer = act.action.__class__._is_peer
        if all(a.name == name and is_peer(a.action.__class__) for a in curr[:idx]):
            del curr[idx]
            return True

    @contextmanager
    def use(self, /, run: bool = False):
        with self.current(self), eager(True):
            yield
            if run:
                self.run()

    def sort(self):
        if not self._sorted:
            super().sort(key=attrgetter("position"))
            self._sorted = True

    def run(self):
        while self:
            self.sort()
            self.pop(0)()
