from collections import OrderedDict
from functools import lru_cache
from hashlib import sha256
from operator import attrgetter
from types import FunctionType, ModuleType
from typing import Any, Callable, Hashable, Iterable, Protocol
from warnings import warn

primitives = (str, int, float, type(None), bytes, complex, type(Ellipsis))

importables = (type, FunctionType, ModuleType)

try:
    from numpy import dtype, ndarray
except ModuleNotFoundError:

    class ndarray: ...  # phony placeholder

else:
    primitives += (dtype,)


primitives_set = frozenset(primitives)


class UsesLocalsError(ValueError): ...


class UsesMainWarning(RuntimeWarning): ...


class UsesPickleWarning(RuntimeWarning): ...


class UnsupportedError(ValueError): ...


class _SHash(list):
    end = sha256(b"").digest()

    def __call__(self, value: Any) -> bytes:
        cls = value.__class__
        if cls in primitives_set:
            return _repr(value)
        if cls is tuple:
            return self.coll(cls, value, self)
        if cls is frozenset:
            return self.coll(cls, value, self, True)
        if isinstance(value, importables):
            return _imp(value)

        try:
            idx = self.index(id(value))
        except ValueError:
            pass
        else:
            return sha256(f"self@{len(self) - idx}".encode()).digest()

        try:
            self.append(id(value))

            if shash := getattr(value, "__shash__", None):
                return shash(self)

            func = self
            sort = False
            if isinstance(value, ndarray):
                value = value.tolist()
            elif isinstance(value, dict):
                sort = not isinstance(value, OrderedDict)
                func = self.item
                value = value.items()
            elif isinstance(value, (list, tuple)):
                pass
            elif isinstance(value, (set, frozenset)):
                sort = True
            elif isinstance(value, (range, slice)):
                value = attrgetter("start", "stop", "step")(value)
            elif isinstance(value, primitives):
                return _repr(value)
            elif callable(red := getattr(value, "__reduce__", None)):
                warn(UsesPickleWarning(cls))
                value = red()
            else:
                raise UnsupportedError(value)

            return self.coll(cls, value, func, sort)
        finally:
            self.pop()

    def item(self, item: tuple[Hashable, Any]) -> bytes:
        h = sha256(self(item[0]))
        h.update(self(item[1]))
        return h.digest()

    def keyh(self, item: tuple[Hashable, bytes]) -> bytes:
        h = sha256(self(item[0]))
        h.update(item[1])
        return h.digest()

    def coll[T](
        self,
        cls: type,
        vals: Iterable[T],
        func: Callable[[T], bytes],
        sort: bool = False,
    ) -> bytes:
        ret = sha256(_imp(cls))
        if sort:
            vals = [func(v) for v in vals]
            vals.sort()
            for val in vals:
                ret.update(val)
        else:
            for val in vals:
                ret.update(func(val))
        ret.update(self.end)
        return ret.digest()


class SHashable(Protocol):
    def __shash__(self, shash: _SHash) -> bytes: ...


def _repr(value) -> bytes:
    return sha256(repr(value).encode()).digest()


@lru_cache
def _imp(value: type | FunctionType) -> bytes:
    if isinstance(value, ModuleType):
        rep = value.__name__
    else:
        rep = f"{value.__module__}:{value.__qualname__}"
    if "<locals>" in rep:
        raise UsesLocalsError(value)
    if rep.startswith("__main__"):
        warn(UsesMainWarning(value))
    return sha256(rep.encode()).digest()


def bin_hash(value: Any) -> bytes:
    return _SHash()(value)


def hex_hash(value: Any) -> str:
    return bin_hash(value).hex()
