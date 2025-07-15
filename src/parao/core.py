from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from math import inf
from types import GenericAlias
from typing import (
    Any,
    Iterable,
    Mapping,
    Protocol,
    Self,
    get_type_hints,
)

from .cast import cast, Opaque

__all__ = ["UNSET", "ParaO", "Param"]


UNSET = Opaque()

_param_counter = count()


def srepr(obj: Any) -> str:
    """get representation, exception safe"""
    try:
        return repr(obj)
    except Exception:
        return object.__repr__(obj)


# type KeyE = type | object | str
type KeyE = str | type | AbstractParam
type KeyT = tuple[KeyE, ...]
type KeyTE = KeyT | KeyE
type TypT = type | GenericAlias
type PrioT = int | float
type Mapish[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]


@dataclass(frozen=True, slots=True)
class Arg:
    key: KeyT
    val: Any
    prio: PrioT = 0
    offset: int = 0

    def __repr__(self):
        parts = list(map(repr, self.key[self.offset :]))
        parts.append(f"val={self.val!r}")
        if self.prio:
            parts.append(f"prio={self.prio!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    __hash__ = object.__hash__  # value can be unhashable!

    @lru_cache
    def solve_value(self, param, owner, name):
        last = len(self.key) - 1
        off = self.offset
        # match owner class filters
        while (
            isinstance((key := self.key[off]), type)
            and off < last
            and issubclass(owner, key)
        ):
            off += 1
        # test param itself
        prio, val, sub = self.prio, UNSET, self
        if param is key or (isinstance(key, str) and key == name):
            if off < last:
                sub = Arg(self.key, self.val, self.prio, off + 1)
            else:
                val = self.val
        return prio, val, sub

    @lru_cache
    def solve_class[T: type](self, cls: T) -> tuple[PrioT, "T | Arg | Arguments"]:
        last = len(self.key) - 1
        off = self.offset
        # match owner class filters
        while (
            off <= last
            and isinstance((key := self.key[off]), type)
            and issubclass(cls, key)
        ):
            off += 1
        if (off == last and key == "__class__") or off > last:
            if not isinstance(self.val, (type, Arg, Arguments)):
                raise TypeError(f"{self!r} resolved to non-class")
            return self.prio, self.val
        return 0, UNSET


class Arguments(tuple["Arguments | Arg", ...]):
    @classmethod
    def make(
        cls, *args: "Arguments | HasArguments | Mapish[KeyTE, Any]", **kwargs: Any
    ):
        return cls._make(args + (kwargs,)) if kwargs else cls._make(args)

    @classmethod
    def _make(cls, args: "tuple[Arguments | HasArguments | Mapish[KeyTE, Any], ...]"):
        sub = []
        if arg := cls._ctxvar.get():
            sub.append(arg)

        for arg in args:
            arg = getattr(arg, "__args__", arg)
            if isinstance(arg, cls):
                if arg:
                    sub.append(arg)
            elif isinstance(arg, dict):
                if arg:
                    sub.append(cls.from_dict(arg.items()))
            else:
                raise TypeError(f"unsupported argument type: {type(arg)}")

        return cls.from_list(sub)

    @classmethod
    def from_dict(
        cls,
        k2v: Mapping[KeyTE, Any] | Iterable[tuple[KeyTE, Any]],
        prio: PrioT = 0,
    ):
        if callable(items := getattr(k2v, "items", None)):
            k2v = items()
        return cls(Arg(k if isinstance(k, tuple) else (k,), v, prio) for k, v in k2v)

    @classmethod
    def from_list(cls, args: "list[Arguments | Arg]") -> "Arguments":
        """Turn an iterable into arguments. Avoids unnecessary nesting or repeated creation of empty Arguments."""
        match args:
            case []:
                return cls.EMPTY
            case [Arguments()]:
                return args[0]
        return cls(args)

    def __repr__(self):
        return self.__class__.__name__ + (tuple.__repr__(self) if self else "()")

    @lru_cache
    def solve_value(self, param, owner, name):
        prio, val, sub = -inf, UNSET, []

        for v in self:
            while isinstance(v, (Arguments, Arg)):
                p, v, s = v.solve_value(param, owner, name)
                if s:
                    sub.append(s)
                if v is UNSET:
                    break
                elif prio <= p:
                    prio = p
                    val = v

        return prio, val, Arguments.from_list(sub)

    @lru_cache
    def solve_class(self, cls):
        prio = -inf

        for v in self:
            while isinstance(v, (Arguments, Arg)):
                p, v = v.solve_class(cls)
                if v is UNSET:
                    break
                elif prio <= p:
                    prio = p
                    cls = v

        return prio, cls


Arguments.EMPTY = Arguments()
Arguments._ctxvar = ContextVar("ContextArguments", default=Arguments.EMPTY)


class HasArguments(Protocol):
    __args__: Arguments


_own_parameters_cache = {}


class ParaOMeta(type):
    @property
    def __own_parameters__(cls) -> "dict[str, AbstractParam]":
        if (val := _own_parameters_cache.get(cls)) is None:
            val = _own_parameters_cache[cls] = {
                name: param
                for name, param in vars(cls).items()
                if not name.startswith("__") and isinstance(param, AbstractParam)
            }
        return val

    def __setattr__(cls, name, value):
        if not name.startswith("__"):
            if cache := _own_parameters_cache.get(cls):
                if old := cache.get(name):
                    old.__set_name__(cls, None)
                    del cache[name]
            if isinstance(value, AbstractParam):
                value.__set_name__(cls, name)
                if cache:
                    cache[name] = value
        return super().__setattr__(name, value)

    def __delattr__(cls, name):
        if not name.startswith("__"):
            if cache := _own_parameters_cache.get(cls):
                if old := cache.get(name):
                    old.__set_name__(cls, None)
                    del cache[name]
        return super().__delattr__(name)

    def __cast_from__(cls, value, original_type):
        if value is UNSET:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(value)

    def __call__(
        cls, *args: Arguments | HasArguments | Mapish[KeyTE, Any], **kwargs: Any
    ) -> Self:
        arg = Arguments._make(args + (kwargs,) if kwargs else args)
        ret = cls.__new__(arg.solve_class(cls)[1])
        ret.__args__ = arg
        ret.__init__()
        return ret


class ParaO(metaclass=ParaOMeta):
    __args__: Arguments  # | UNSET


class ParamAlias(GenericAlias):
    def __call__(self, *args, **kwds):
        (typ,) = self.__args__
        kwds.setdefault("type", typ)
        return super().__call__(*args, **kwds)


### actual code
class AbstractParam[T]:
    min_prio = -inf

    def __class_getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != 1:
            raise TypeError(f"{cls.__qualname__} can only receive one type argument")
        return ParamAlias(cls, key)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._owner2name = {}
        self._id = next(_param_counter)

    def __set_name__(self, cls, name):
        if name:
            self._owner2name[cls] = name
        else:
            del self._owner2name[cls]
        self._solve_name.cache_clear()

    @staticmethod
    @lru_cache
    def _solve_name(param: "Param", icls: "ParaOMeta") -> str | None:
        lut = param._owner2name
        for cls in icls.__mro__:
            if cls in lut:
                return lut[cls]

    @staticmethod
    @lru_cache
    def _solve_types(cls: "ParaOMeta"):
        return get_type_hints(cls)

    def _get(self, val: Any, name: str, instance: "ParaO") -> T:
        typ = self.type
        if typ is UNSET:
            typ = self._solve_types(type(instance)).get(name, UNSET)
        return cast(val, typ)

    def __get__(self, instance: "ParaO", owner: type | None = None) -> T:
        if instance is None:
            return self
        name = self._solve_name(self, type(instance))

        # prio, val, sub = instance.__args__._solve_arguments(self, type(instance), name)
        prio, val, sub = instance.__args__.solve_value(self, type(instance), name)

        try:
            tok = Arguments._ctxvar.set(sub)
            val = self._get(val, name, instance)
        except Exception as e:
            e.add_note(f"parameter {name}={srepr(self)} on {srepr(instance)}")
            raise
        finally:
            Arguments._ctxvar.reset(tok)

        instance.__dict__[name] = val
        return val

    type: type


AbstractParam.type = UNSET


class MissingParameterValue(TypeError): ...


class Param[T](AbstractParam[T]):
    def __init__(self, default=UNSET, **kwargs):
        super().__init__(default=default, **kwargs)

    def _get(self, val, name, instance):
        val = super()._get(val, name, instance)
        if val is UNSET:
            if self.default is UNSET:
                raise MissingParameterValue(name)
            else:
                return self.default
        return val
