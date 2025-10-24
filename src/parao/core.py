from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, lru_cache, partial
from itertools import count
from math import inf
from os.path import dirname
from types import GenericAlias
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Iterable,
    Mapping,
    Protocol,
    Self,
    TypeVar,
    get_type_hints,
    overload,
)
from warnings import warn
from weakref import WeakKeyDictionary

from .cast import Opaque, cast
from .misc import ContextValue, safe_len, safe_repr
from .shash import _SHash, bin_hash

__all__ = ["UNSET", "ParaO", "Param", "Prop", "Const"]
_warn_skip = (dirname(__file__),)


class Unset(Opaque): ...


UNSET = Unset()
UNSET.__shash = _SHash().coll(Unset, (), None)
UNSET.__shash__ = lambda _: UNSET.__shash

Unset.__new__ = lambda _: UNSET

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

    @property
    def effective_key(self):
        return self.key[self.offset :]

    def __gt__(self, other: "Arg | None") -> bool:
        return other is None or self.prio > other.prio

    def __lt__(self, other: "Arg | None") -> bool:
        return other is not None and self.prio < other.prio

    # @lru_cache
    def _solve_value(self, ref: "ParaOMeta"):
        last = len(self.key) - 1
        off = self.offset
        # match owner class filters
        while (
            isinstance((key := self.key[off]), type)
            and off < last
            and issubclass(ref, key)
        ):
            off += 1

        op = ref.__own_parameters__
        if key in op.vals or (key := op.get(key)):
            return (
                key,
                Arg(self.key, self.val, self.prio, off + 1),
                off == self.offset,  # is_final & no class filter
            )
        else:
            return None, None, False

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

    def get_root_of(self, arg: "Arg") -> "Arg | None":
        if (
            self.key is arg.key
            and self.val is arg.val
            and self.prio is arg.prio
            and self.offset <= arg.offset
        ):
            return self


class Arguments(tuple["Arguments | Arg", ...]):
    is_final = False

    @classmethod
    def make(cls, *args: "Arguments | HasArguments | dict[KeyTE, Any]", **kwargs: Any):
        return cls._make(args + (kwargs,)) if kwargs else cls._make(args)

    @classmethod
    def _make(cls, args: "tuple[Arguments | HasArguments | dict[KeyTE, Any], ...]"):
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

    def solve_value(self, param, owner, name) -> tuple[Arg | None, "Arguments"]:
        com, val, sub = self._solve_values(owner)
        su = sub.get(param)

        return val.get(param), com if su is None else Arguments.from_list(su)

    @lru_cache
    def _solve_values(self, ref: "ParaOMeta"):
        com = []
        val: dict[AbstractParam, Arg] = {}
        sub: defaultdict[AbstractParam, list[Arguments | Arg]]

        if False:
            sub = defaultdict(com.copy)  # TODO: optimize this
        else:  # optimized behaviour

            @defaultdict
            def sub():
                nonlocal com
                if len(com) > 1:
                    com = [Arguments.from_list(com)]
                return com.copy()

        for arg in self:
            if isinstance(arg, Arg):
                k, d, r = arg._solve_value(ref)
                if not r:
                    sub.get(k, com).append(arg)
                if d is not None:
                    assert k is not None
                    if d.is_final:
                        if isinstance(d.val, (Arguments, Arg)):
                            sub[k].append(d.val)
                        else:
                            val[k] = max(d, val.get(k))
                    else:
                        sub[k].append(d)
            else:
                c, v, s = arg._solve_values(ref)
                val.update(v)
                for k, v in s.items():
                    sub[k].extend(v)
                com.append(c)  # must be done after filling sub

        return Arguments.from_list(com), val, sub

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

    def get_root_of(self, arg: "Arg") -> "Arg | None":
        for a in self:
            if r := a.get_root_of(arg):
                return r


Arguments.EMPTY = Arguments()
Arguments._ctxargs = ContextValue("ContextArguments", default=Arguments.EMPTY)


class HasArguments(Protocol):
    __args__: Arguments


eager = ContextValue[bool]("eager", default=False)


class _OwnParameters(dict[str, "AbstractParam"]):
    __slots__ = "vals"
    vals: set["AbstractParam"]

    def __init__(self, cls: "ParaOMeta"):
        super().__init__(
            (name, param)
            for name in dir(cls)
            if not name.startswith("__")
            and isinstance((param := getattr(cls, name)), AbstractParam)
        )
        self.vals = set(self.values())

    cache: dict["ParaOMeta", "_OwnParameters"] = {}


class ParaOMeta(type):
    @property
    def __fullname__(cls):
        return f"{cls.__module__}:{cls.__qualname__}"

    @property
    def __own_parameters__(cls) -> _OwnParameters:
        if (val := _OwnParameters.cache.get(cls)) is None:
            val = _OwnParameters.cache[cls] = _OwnParameters(cls)
        return val

    def __setattr__(cls, name, value):
        if not name.startswith("__"):
            if cache := _OwnParameters.cache.get(cls):
                if old := cache.get(name):
                    old.__set_name__(cls, None)
                    cache.vals.remove(old)
                    del cache[name]
            if isinstance(value, AbstractParam):
                value.__set_name__(cls, name)
                if cache:
                    cache.vals.add(value)
                    cache[name] = value
        return super().__setattr__(name, value)

    def __delattr__(cls, name):
        if not name.startswith("__"):
            if cache := _OwnParameters.cache.get(cls):
                if old := cache.get(name):
                    old.__set_name__(cls, None)
                    cache.vals.remove(old)
                    del cache[name]
        return super().__delattr__(name)

    def __cast_from__(cls, value, original_type):
        if value is UNSET:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(value)

    def __call__(
        cls, *args: Arguments | HasArguments | dict[KeyTE, Any], **kwargs: Any
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

    def __shash__(self, enc: _SHash) -> bytes:
        if (res := getattr(self, "__shash", None)) is None:
            res = self.__shash = enc.coll(
                self.__class__,
                (
                    (name, vhash)
                    for name, param in self.__class__.__own_parameters__.items()
                    if param.significant
                    and (value := getattr(self, name)) is not param.neutral
                    and (vhash := enc(value)) != enc(param.neutral)
                ),
                enc.keyh,
                True,
            )
        return res

    def __hash__(self) -> int:
        return int.from_bytes(bin_hash(self)[:8])

    @cached_property
    def __inner__(self) -> tuple["ParaO", ...]:
        ret = []
        for name, param in self.__class__.__own_parameters__.items():
            if param.significant:
                val = getattr(self, name)
                if isinstance(val, ParaO):
                    ret.append(val)
                elif isinstance(val, Expansion):
                    ret.extend(val.expand())
                elif isinstance(val, (dict, list, tuple, set, frozenset)):
                    queue = [iter((val,))]
                    while queue:
                        for curr in queue[-1]:
                            if isinstance(curr, (list, tuple, set, frozenset)):
                                queue.append(iter(curr))
                                break
                            elif isinstance(curr, dict):
                                queue.append(iter(curr.keys()))
                                queue.append(iter(curr.values()))
                                break
                            elif isinstance(curr, ParaO):
                                ret.append(curr)
                        else:
                            queue.pop()
        return tuple(ret)

    def __repr__(self, *, compact: bool | str = False):
        if compact:
            if compact is True:
                compact = "..."
            items = [compact]
        else:
            items = [
                f"{name}={value!r}"
                for name, value, neutral in self.__rich_repr__()
                if value != neutral
            ]

        return f"{self.__class__.__fullname__}({", ".join(items)})"

    def __rich_repr__(self):
        for name, param in self.__class__.__own_parameters__.items():
            if param.significant and not isinstance(param, Const):
                if (neutral := param.neutral) is UNSET:
                    neutral = getattr(param, "default", UNSET)
                yield name, getattr(self, name), neutral


class TypedAlias(GenericAlias):
    _typevar2name = WeakKeyDictionary[TypeVar, str]()  # shadowed on instances!

    class TypedAliasMismatch(RuntimeWarning): ...

    class TypedAliasClash(TypeError): ...

    class TypedAliasRedefined(RuntimeWarning): ...

    def __init__(self, *arg, **kwargs):
        super().__init__()
        cls = self.__class__
        tv2n = cls._typevar2name
        for arg, tp in zip(self.__args__, self.__origin__.__type_params__):
            if name := tv2n.get(tp):
                if isinstance(arg, TypeVar):
                    if arg.__name__ != tp.__name__:
                        warn(f"{arg} -> {tp}", cls.TypedAliasMismatch, stacklevel=4)
                    cls.register(arg, name)

    def __call__(self, *args, **kwds):
        tv2n = self.__class__._typevar2name
        for arg, tp in zip(self.__args__, self.__origin__.__type_params__):
            if name := tv2n.get(tp):
                if not isinstance(arg, TypeVar):
                    kwds.setdefault(name, arg)
        return super().__call__(*args, **kwds)

    @classmethod
    def convert(cls, ga: GenericAlias):
        return cls(ga.__origin__, ga.__args__)

    @classmethod
    def register(cls, tv: TypeVar, name: str):
        if got := cls._typevar2name.get(tv):
            if got != name:
                raise cls.TypedAliasClash(f"{tv} wants {name!r} already got {got!r}")
            else:
                warn(str(tv), cls.TypedAliasRedefined, skip_file_prefixes=_warn_skip)
        else:
            cls._typevar2name[tv] = name

    @classmethod
    def init_subclass(cls, subcls: "type[AbstractParam]"):
        for ob in reversed(subcls.__orig_bases__):
            if isinstance(ob, cls):
                for arg, tp in zip(ob.__args__, ob.__origin__.__type_params__):
                    if name := cls._typevar2name.get(tp):
                        if not isinstance(arg, TypeVar) and not hasattr(subcls, name):
                            setattr(subcls, name, arg)


class UntypedWarning(RuntimeWarning):
    @classmethod
    def warn(cls, param: "AbstractParam", instance: "ParaO", name: str | None = None):
        if name is None:
            name = param._name(type(instance))
        warn(
            f"{type(param)} {name} on {type(instance)}",
            category=cls,
            skip_file_prefixes=_warn_skip,
        )


class UntypedParameter(UntypedWarning): ...


type ExpansionFilter = Collection[KeyE | Collection[KeyE]] | Callable[
    [Expansion, ParaO], bool
]


class DuplicateParameter(RuntimeError): ...


@lru_cache
def _solve_name(param: "Param", icls: "ParaOMeta") -> str | None:
    lut = param._owner2name
    for cls in icls.__mro__:
        if cls in lut:
            return lut[cls]


@lru_cache
def _get_type_hints(cls: "ParaOMeta"):
    return get_type_hints(cls)


### actual code
class AbstractParam[T]:
    significant: bool = True
    neutral: T = UNSET

    TypedAlias.register(T, "type")

    def __class_getitem__(cls, key):
        return TypedAlias.convert(super().__class_getitem__(key))

    def __init_subclass__(cls):
        super().__init_subclass__()
        TypedAlias.init_subclass(cls)

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
        _solve_name.cache_clear()

    def _name(self, cls: "ParaOMeta"):
        return _solve_name(self, cls)

    def _type(self, cls: "ParaOMeta", name: str):
        typ = self.type
        if typ is UNSET:
            typ = _get_type_hints(cls).get(name, UNSET)
        return typ

    def _cast(self, val, typ):

        if typ is not UNSET:
            try:
                exp = cast(val, Expansion[typ])
            except TypeError:
                pass
            else:
                if isinstance(exp, Expansion):
                    raise exp
        return cast(val, typ)

    def _get(self, val: Any, name: str, instance: "ParaO") -> T:
        typ = self._type(type(instance), name)
        if typ is UNSET:
            UntypedParameter.warn(self, instance, name)
            return val
        return self._cast(val, typ)

    def _collect(self, expansion: "Expansion", instance: "ParaO"):
        return bool(self.collect) and (
            self.collect(expansion, instance)
            if callable(self.collect)
            else any(map(expansion.test, self.collect))
        )

    def _solve(
        self,
        arg: Arg | None,
        name: str,
        instance: "ParaO",
        *args: Arguments | Arg,
    ):
        val = UNSET if arg is None or arg.prio < self.min_prio else arg.val

        try:
            with Arguments._ctxargs(Arguments.from_list(args)):
                return self._get(val, name, instance)
        except Expansion as exp:
            exp.process(self, instance, arg)
            exp.make = partial(self._solve, arg, name, instance, *args)
            return exp
        except Exception as exc:
            exc.add_note(f"parameter {name}={safe_repr(self)} on {safe_repr(instance)}")
            raise

    def __get__(self, instance: "ParaO", owner: type | None = None) -> T:
        if instance is None:
            return self
        cls = type(instance)
        name = self._name(cls)

        arg, sub = instance.__args__.solve_value(self, cls, name)

        instance.__dict__[name] = val = self._solve(arg, name, instance, sub)
        return val

    min_prio: float = -inf
    eager: bool = True
    collect: ExpansionFilter = ()
    type: type | Unset
    type = UNSET


class Const[T](AbstractParam[T]):
    def __init__(self, value: T, **kwargs):
        super().__init__(value=value, **kwargs)

    def __get__(self, instance, owner=None) -> T:
        return self if instance is None else self.value


class AbstractDecoParam[T, F: Callable](AbstractParam[T]):
    def __init__(self, func: F, **kwargs):
        assert callable(func)
        super().__init__(func=func, **kwargs)

    @overload
    def __new__(cls: Self, func: None = None, **kwargs) -> partial[Self]: ...

    @overload
    def __new__(cls: Self, func: F = None, **kwargs) -> Self: ...

    def __new__(cls, func: F | None = None, **kwargs):
        if func is None:
            return partial(cls, **kwargs)
        return super().__new__(cls)

    def __getattr__(self, name):
        if not name.startswith("_"):
            return getattr(self.func, name)

    func: F


class Prop[T](AbstractDecoParam[T, Callable[[ParaO], T]]):
    def _type(self, cls, name):
        typ = getattr(self.func, "__annotations__", {}).get("return", UNSET)
        if typ is UNSET:
            typ = super()._type(cls, name)
        return typ

    def _get(self, val, name, instance) -> T:
        val = super()._get(val, name, instance)
        return self.func(instance) if val is UNSET else val


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
        assert iter(values)  # ensure values is iterable
        self.values = values
        self._frames: list[tuple[ParaOMeta, Param, Arg | None]] = []

    make: Callable[[Arg], T | Self] | None = None

    def test(self, item: KeyE | Iterable[KeyE]):
        match item:
            case AbstractParam():
                return self.param is item
            case str():
                return bool(self.param_name == item)
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
        if param._collect(self, inst):
            return  # this will add ._get
        # keep track of key to dial-down to the origin
        self._frames.append(
            (inst.__class__, param, inst.__args__.get_root_of(self.arg))
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
        for cls, param, root in self._frames:
            if use_arg and root is not None:
                rkey = list(root.effective_key[::-1])
                continue

            name = _solve_name(param, cls)

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

    def expand(self, prio: PrioT = 0, **kwargs) -> Generator[T]:
        key = self.make_key(**kwargs)
        for val in self.values:
            res = self.make(Arg(key, val, prio))
            if isinstance(res, Expansion):
                yield from res.expand(prio=prio, **kwargs)
            else:
                yield res

    @staticmethod
    def generate(typ: ParaOMeta, args: Arguments, **kwargs) -> Generator[ParaO]:
        try:
            yield typ(args)
        except Expansion as exp:
            exp.make = lambda arg: typ(Arguments((args, arg)))
            try:
                yield from exp.expand(**kwargs)
            except Exception as exc:
                exc.add_note(f"while expanding: {exp!r}")
                raise

    @property
    def param_name(self):
        return self.param._name(type(self.source))

    def __repr__(self):
        parts = [f"< {safe_len(self.values)} values >"]
        if self._frames:
            parts.append(f"param={self.param}")
            parts.append(f"source={self.source.__repr__(compact=True)}")
        return f"{self.__class__.__name__}({', '.join(parts)})"
