from contextlib import contextmanager
from functools import lru_cache, partial
from inspect import Parameter, signature
from typing import Any, Callable, Concatenate, Iterable, Type

from .core import (
    UNSET,
    AbstractDecoParam,
    Expansion,
    MissingParameterValue,
    ParaO,
    Unset,
    eager,
)
from .misc import ContextValue


@lru_cache
def _method_1st_arg_annotation[T](
    func: Callable[Concatenate[Any, T, ...], Any],
) -> Type[T] | Unset:
    for i, param in enumerate(signature(func).parameters.values()):
        if i == 1:
            if (
                param.kind
                in (
                    Parameter.POSITIONAL_ONLY,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    Parameter.VAR_POSITIONAL,
                )
                and param.annotation is not Parameter.empty
            ):
                return param.annotation
            break
    return UNSET


class Act[T, R](partial):
    func: "Action[T, R]"
    args: tuple[ParaO, T]


class Action[T, R, **Ps](AbstractDecoParam[T, Callable[Concatenate[ParaO, Ps], R]]):
    _act_cls: type[Act] = Act
    significant = False

    def _get(self, val, name, instance) -> Act[T, R]:
        val = super()._get(val, name, instance)
        act = self._act_cls(self, instance, val)
        if val is not UNSET:
            Plan.add(act)
        return act

    __get__: Callable[..., Act[T, R] | Expansion]

    def __call__(
        self, inst: ParaO, value: T | Unset, *args: Ps.args, **kwargs: Ps.kwargs
    ):
        if value is UNSET:
            return self.func(inst, *args, **kwargs)
        else:
            return self.func(inst, value, *args, **kwargs)


class MissingParameterValueOrOverride(MissingParameterValue): ...


class ValueAction[T, R](Action[T, R, T]):
    def _type(self, cls, name):
        typ = self.type
        if typ is UNSET:
            typ = _method_1st_arg_annotation(self.func)
        return typ

    __get__: Callable[..., Act[T, R] | Expansion]

    def __call__(self, inst: ParaO, value: T | Unset, override: T | Unset = UNSET):
        if override is not UNSET:
            value = override
        if value is UNSET:
            raise MissingParameterValueOrOverride(self._name(type(inst)))
        return self.func(inst, value)


class RecursiveAction(Action[int | bool | None, None, [int, Iterable[Act]]]):
    func: Callable[[ParaO, int, Iterable[Act]], Iterable[Act] | None]
    _func: Callable[[ParaO, int, Iterable[Act]], Iterable[Act] | None] | None = None

    def _solve_inner(self, inst: ParaO) -> Iterable[Act]:
        name = self._name(type(inst))
        cls = type(self)
        for inner in inst.__inner__:
            if other := inner.__class__.__own_parameters__.get(name):
                if isinstance(other, cls):
                    yield getattr(inner, name)

    def __call__(
        self,
        inst: ParaO,
        value: int | bool | Unset = UNSET,
        override: int | bool | Unset = UNSET,
        *,
        outer: int = True,
        depth: int = 0,
    ):
        if override is not UNSET:
            value = override
        if value is UNSET:
            value = outer
        elif value is False or value < 0:
            return

        inner = self._solve_inner(inst) if value else ()

        _func = self._func or self.func
        ret = _func(inst, depth, inner)
        if ret is None:
            ret = inner

        kwargs = dict(depth=depth + 1)
        if value is not True and value > 0:
            kwargs["outer"] = value - 1

        for r in ret:
            r(**kwargs)


class Plan(list[Act]):
    current = ContextValue["Plan"]("currentActionPlan", default=None)

    @classmethod
    def add(cls, act: Act):
        if (curr := cls.current()) is not None:
            curr.append(act)

    @contextmanager
    def use(self, /, run: bool = False):
        with self.current(self), eager(True):
            yield
            if run:
                self.run()

    def run(self):
        while self:
            self.pop(0)()
