from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from functools import partial
from os.path import dirname
from types import GenericAlias
from typing import Any, Self, TypeVar, overload
from warnings import warn
from weakref import WeakKeyDictionary

_sentinel = object()

__all__ = ["ContextValue", "context_manager_set", "safe_repr", "safe_len"]

ewarn = partial(warn, skip_file_prefixes=(dirname(__file__),))


@contextmanager
def context_manager_set[T](contextvar: ContextVar[T], value: T):
    token = contextvar.set(value)
    try:
        yield token.old_value
    finally:
        contextvar.reset(token)


class ContextValue[T]:
    __slots__ = ("contextvar",)

    def __init__(self, name: str, *, default: T = _sentinel):
        if default is _sentinel:
            self.contextvar = ContextVar[T](name)
        else:
            self.contextvar = ContextVar[T](name, default=default)

    @overload
    def __call__(self) -> T:
        """Return the current context variable value or the global default, if any."""

    @overload
    def __call__(self, *, default: T) -> T:
        """Return the current context variable value or the given default."""

    @overload
    def __call__(self, value: T) -> AbstractContextManager[T, None]:
        """Set the context variable value to the given value and returns a context manager that passes the previous value and reset the value upon exit."""

    def __call__(self, value: T = _sentinel, *, default: T = _sentinel):
        if value is _sentinel:
            if default is _sentinel:
                return self.contextvar.get()
            else:
                return self.contextvar.get(default)
        else:
            return context_manager_set(self.contextvar, value)


def safe_repr(obj: Any) -> str:
    """get representation, exception safe"""
    try:
        return repr(obj)
    except Exception:
        return object.__repr__(obj)


def safe_len[T](obj: Any, default: T = None) -> int | T:
    """get len of obj, return default on failure"""
    try:
        return len(obj)
    except Exception:
        return default


class PeekableIter[T]:
    __slots__ = (
        "_iter",
        "_head",
    )

    def __init__(self, it: Iterable[T]):
        self._iter = iter(it)
        self._head = _sentinel

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self._head is _sentinel:
            return next(self._iter)
        else:
            ret = self._head
            self._head = _sentinel
            return ret

    def peek(self, default=_sentinel) -> T:
        if self._head is _sentinel:
            try:
                self._head = next(self._iter)
            except StopIteration:
                if default is _sentinel:
                    raise
                else:
                    return default
        return self._head

    @property
    def more(self) -> bool:
        if self._head is _sentinel:
            try:
                self._head = next(self._iter)
            except StopIteration:
                return False
        return True


def is_subseq(needles, haystack):
    haystack = iter(haystack)
    return all(needle in haystack for needle in needles)


class StrOpBuffer(list[str]):
    __slots__ = ("func",)

    def __init__(self, func: Callable[[Self], str]):
        super().__init__()
        self.func = func

    def flush(self):
        ret = self.func(self)
        self.clear()
        return ret


class TypedAliasMismatch(RuntimeWarning): ...


class TypedAliasClash(TypeError): ...


class TypedAliasRedefined(RuntimeWarning): ...


class TypedAlias(GenericAlias):
    _typevar2name = WeakKeyDictionary[TypeVar, str]()  # shadowed on instances!

    def __init__(self, *_):
        super().__init__()
        cls = self.__class__
        tv2n = cls._typevar2name
        for arg, tp in zip(self.__args__, self.__origin__.__type_params__):
            if name := tv2n.get(tp):
                if isinstance(arg, TypeVar):
                    if arg.__name__ != tp.__name__:
                        warn(f"{arg} -> {tp}", TypedAliasMismatch, stacklevel=4)
                    cls.register(arg, name)

    def __call__(self, *args, **kwds):
        tv2n = self.__class__._typevar2name
        for arg, tp in zip(self.__args__, self.__origin__.__type_params__):
            if name := tv2n.get(tp):
                assert not isinstance(arg, TypeVar)  # already registered during init
                kwds.setdefault(name, arg)
        return super().__call__(*args, **kwds)

    @classmethod
    def convert(cls, ga: GenericAlias):
        return cls(ga.__origin__, ga.__args__)

    @classmethod
    def register(cls, tv: TypeVar, name: str):
        if got := cls._typevar2name.get(tv):
            if got != name:
                raise TypedAliasClash(f"{tv} wants {name!r} already got {got!r}")
            else:
                ewarn(str(tv), TypedAliasRedefined)
        else:
            cls._typevar2name[tv] = name

    @classmethod
    def init_subclass(cls, subcls: type):
        for ob in reversed(subcls.__orig_bases__):
            if isinstance(ob, cls):
                for arg, tp in zip(ob.__args__, ob.__origin__.__type_params__):
                    if name := cls._typevar2name.get(tp):
                        if not isinstance(arg, TypeVar) and not hasattr(subcls, name):
                            setattr(subcls, name, arg)
