from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import count
from math import inf
from types import GenericAlias
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Protocol,
    Self,
    get_type_hints,
)
from warnings import warn
from os.path import dirname

from .cast import cast, Opaque
from .misc import ContextValue, safe_len, safe_repr

__all__ = ["UNSET", "ParaO", "Param"]
_warn_skip = (dirname(__file__),)

UNSET = Opaque()

_param_counter = count()


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

    @property
    def is_final(self):
        return self.offset >= len(self.key)

    def __gt__(self, other: "Arg | None") -> bool:
        return other is None or self.prio > other.prio

    def __lt__(self, other: "Arg | None") -> bool:
        return other is not None and self.prio < other.prio

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
        if param is key or (isinstance(key, str) and key == name):
            arg = Arg(self.key, self.val, self.prio, off + 1)
            if off < last:
                return None, arg
            else:
                return arg, ()
        return None, self

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

    def has_root(self, arg: "Arg") -> bool:
        return (
            self.key is arg.key
            and self.val is arg.val
            and self.prio is arg.prio
            and self.offset <= arg.offset
        )


class Arguments(tuple["Arguments | Arg", ...]):
    is_final = False

    @classmethod
    def make(
        cls, *args: "Arguments | HasArguments | Mapish[KeyTE, Any]", **kwargs: Any
    ):
        return cls._make(args + (kwargs,)) if kwargs else cls._make(args)

    @classmethod
    def _make(cls, args: "tuple[Arguments | HasArguments | Mapish[KeyTE, Any], ...]"):
        sub = []
        if arg := cls._ctxargs():
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
    def solve_value(self, param, owner, name) -> tuple[Arg | None, "Arguments"]:
        val, sub = None, []

        for v in self:
            while True:
                v, s = v.solve_value(param, owner, name)
                if s:
                    sub.append(s)
                if v is None:
                    break
                else:
                    val = max(v, val)
                    if v.is_final:
                        break

        return val, Arguments.from_list(sub)

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

    def has_root(self, arg: "Arg"):
        return any(v.has_root(arg) for v in self)


Arguments.EMPTY = Arguments()
Arguments._ctxargs = ContextValue("ContextArguments", default=Arguments.EMPTY)


class HasArguments(Protocol):
    __args__: Arguments


eager = ContextValue[bool]("eager", default=False)

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
        if eager():
            for name, param in ret.__class__.__own_parameters__.items():
                if param.eager:
                    getattr(ret, name)
        return ret


class ParaO(metaclass=ParaOMeta):
    __args__: Arguments  # | UNSET


class TypedAlias(GenericAlias):
    def __call__(self, *args, **kwds):
        (typ,) = self.__args__
        kwds.setdefault("type", typ)
        return super().__call__(*args, **kwds)


class UntypedParameter(RuntimeWarning): ...


type ExpansionFilter = Collection[KeyE | Collection[KeyE]] | Callable[
    [Expansion, ParaO], bool
]


class DuplicateParameter(RuntimeError): ...


### actual code
class AbstractParam[T]:
    def __class_getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != 1:
            raise TypeError(f"{cls.__qualname__} can only receive one type argument")
        return TypedAlias(cls, key)

    __slots__ = ("__dict__", "_owner2name", "_id")

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._owner2name = {}
        self._id = next(_param_counter)

    def __set_name__(self, cls, name):
        if name:
            old = self._owner2name.get(cls, None)
            if old is not None:
                raise DuplicateParameter(f"{self} on {cls} with names: {old}, {name}")
            self._owner2name[cls] = name
        else:
            del self._owner2name[cls]
        self._solve_name.cache_clear()

    def solve_name(self, instance: "ParaO"):
        return self._solve_name(self, type(instance))

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

    def _cast(self, val, typ):
        if True or typ is not UNSET:
            try:
                exp = cast(val, Expansion[typ])
            except Exception:
                pass
            else:
                if isinstance(exp, Expansion):
                    raise exp
        return cast(val, typ)

    def _get(self, val: Any, name: str, instance: "ParaO") -> T:
        typ = self.type
        if typ is UNSET:
            typ = self._solve_types(type(instance)).get(name, UNSET)
        if typ is UNSET:
            warn(
                f"{type(self)} {name} on {type(instance)}",
                category=UntypedParameter,
                skip_file_prefixes=_warn_skip,
            )
            return val
        return self._cast(val, typ)

    def __get(
        self,
        arg: Arg | None,
        name: str,
        instance: "ParaO",
        *args: Arguments | Arg,
    ) -> T:
        val = UNSET if arg is None or arg.prio < self.min_prio else arg.val

        try:
            with Arguments._ctxargs(Arguments.from_list(args)):
                return self._get(val, name, instance)
        except Expansion as exp:
            exp.process(self, instance, arg)
            exp._get = partial(self.__get, arg, name, instance, *args)
            return exp
        except Exception as exc:
            exc.add_note(f"parameter {name}={safe_repr(self)} on {safe_repr(instance)}")
            raise

    def __get__(self, instance: "ParaO", owner: type | None = None) -> T:
        if instance is None:
            return self
        name = self._solve_name(self, type(instance))

        arg, sub = instance.__args__.solve_value(self, type(instance), name)

        instance.__dict__[name] = val = self.__get(arg, name, instance, sub)
        return val

    min_prio: float = -inf
    eager: bool = True
    collect: ExpansionFilter = ()
    type: type  # | UNSET
    type = UNSET


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


class ExpansionGeneratedKeyMissingParameter(RuntimeWarning): ...


class Expansion[T](BaseException):
    @classmethod
    def __cast_from__(cls, value, original_type):
        (typ,) = original_type.__args__
        result = cast(value, tuple[typ, ...])
        if isinstance(result, tuple):
            return cls(result)
        else:
            raise NotImplementedError

    def __init__(self, values: Iterable[T]):
        super().__init__()
        iter(values)  # ensure values is iterable
        self.values = values
        self._frames = []

    def test(self, item: KeyE | Iterable[KeyE]):
        match item:
            case AbstractParam():
                return self.param is item
            case str():
                return self.param_name == item
            case type():
                return isinstance(self.source, item)
            case _:
                return all(map(self.test, item))

    def process(self, param: AbstractParam, inst: ParaO, arg: Arg):
        # leave marks of origin
        if not self._frames:
            self.param = param
            self.source = inst
            # do we need these? or are they only for _unwind construction
            self.arg = arg
        # is it collected here?
        if param.collect and (
            param.collect(self, inst)
            if callable(param.collect)
            else any(map(self.test, param.collect))
        ):
            return  # this will add ._get
        # keep track of key to dial-down to the origin
        self._frames.append(
            (
                inst.__class__,
                param,
                inst.__args__.has_root(self.arg),
            )
        )
        raise  # self # but don't to avoid mangling the traceback

    def make_key(
        self,
        use_arg: bool = True,
        *,
        dont: Collection[KeyE] = (),
        use_cls: bool = True,
        use_param: bool = True,
        use_name: bool = True,
        want: Collection[KeyE] | None = None,
    ):
        dont = set(dont)
        dont.difference_update(
            dont_cls := tuple(d for d in dont if isinstance(d, type))
        )

        if want is not None:
            want = set(want)
            want.difference_update(
                want_cls := tuple(w for w in want if isinstance(w, type))
            )

        rkey = []
        if use_arg:
            rkey.extend(reversed(self.arg.key))
        for cls, param, has_arg_root in self._frames:
            if use_arg:
                if has_arg_root:
                    continue
                else:
                    use_arg = False

            name = param._solve_name(param, cls)

            if (
                use_param
                and param not in dont
                and name not in dont
                and (want is None or param in want or name in want)
            ):
                rkey.append(name if use_name else param)
            if (
                use_cls
                and not issubclass(cls, dont_cls)
                and (want is None or issubclass(cls, want_cls))
            ):
                if not rkey:
                    rkey.append(name if use_name else param)
                    warn(
                        f"force added {rkey[0]!r}",
                        ExpansionGeneratedKeyMissingParameter,
                    )
                rkey.append(cls)
        return tuple(reversed(rkey))

    def expand(self, prio: PrioT = 0, **kwargs):
        key = self.make_key(**kwargs)
        for val in self.values:
            res = self._get(Arg(key, val, prio))
            if isinstance(res, Expansion):
                yield from res.expand(prio=prio, **kwargs)
            else:
                yield res

    @property
    def param_name(self):
        return self.param.solve_name(self.source)

    def __repr__(self):
        parts = [f"< {safe_len(self.values)} values >"]
        if self.param:
            parts.append(f"param={self.param}")
        if self.source:
            parts.append(f"source={self.source}")
        return f"{self.__class__.__name__}({', '.join(parts)})"
