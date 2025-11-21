from abc import ABCMeta
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import count
from math import inf
from pickle import PicklingError
from types import GenericAlias
from typing import Any, Protocol, Self, get_type_hints, overload
from warnings import catch_warnings
from weakref import WeakSet

from .cast import Opaque, cast
from .misc import ContextValue, TypedAlias, TypedAliasRedefined, ewarn, safe_len
from .shash import _SHash, bin_hash

__all__ = ["UNSET", "ParaO", "Param", "Prop", "Const"]


class Unset(Opaque):
    def __repr__(self):
        return "<UNSET>"


UNSET = Unset()
UNSET.__shash = _SHash().coll(Unset, (), None)
UNSET.__shash__ = lambda _: UNSET.__shash

Unset.__new__ = lambda _: UNSET

_param_counter = count()


# type KeyE = type | object | str
type KeyE = str | type | _Param
type KeyT = tuple[KeyE, ...]
type KeyTE = KeyT | KeyE
type TypT = type | GenericAlias
type PrioT = int | float
type Mapish[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]


@dataclass(slots=True, frozen=True, weakref_slot=True)
class Value[T: Any]:
    val: T
    prio: PrioT = 0
    position: int = 0

    used = WeakSet["Value"]()
    seen = WeakSet["Value"]()

    def __hash__(self):
        return hash((self.__class__, id(self.val), self.prio, self.position))

    def __eq__(self, other):
        return (
            isinstance(other, Value)
            and self.val is other.val
            and self.prio == other.prio
            and self.position == other.position
        )

    def __or__(self, other: "Value | None"):
        return self if other is None or self.prio > other.prio else other

    def __ror__(self, other: "Value | None"):
        return self if other is None or self.prio >= other.prio else other

    def __repr__(self):
        ret = [repr(self.val)]
        if self.prio or self.position:
            ret.append(repr(self.prio))
        if self.position:
            ret.append(repr(self.position))
        return f"{self.__class__.__name__}({', '.join(ret)})"


@dataclass(slots=True, frozen=True)
class Fragment:
    param: "str | _Param | None"
    types: tuple[type] | None
    inner: "Fragment | Arguments | Value"

    @classmethod
    def make(cls, key: KeyTE, value: "Value | Fragment | Arguments"):
        assert value is None or isinstance(value, (Value, Fragment, Arguments))
        if not isinstance(key, tuple):
            key = (key,)
        return cls._make(iter(key), value)

    @classmethod
    def _make(
        cls, it: Iterator[KeyE], value: "Value | Fragment | Arguments"
    ) -> "Fragment":
        types = []
        for k in it:
            if isinstance(k, type):
                types.append(k)
            elif isinstance(k, (str, _Param)):
                return cls(k, tuple(types) if types else None, cls._make(it, value))
            else:
                raise TypeError(f"invaid key components: {k!r}")
        else:
            return cls(None, tuple(types), value) if types else value

    def is_type_ok(self, ref: type):
        if typs := self.types:
            return all(issubclass(ref, typ) for typ in typs)
        return True

    @property
    def effective_key(self) -> KeyT:
        assert self.param is not None
        r = (self.types or ()) + (self.param,)
        if isinstance(i := self.inner, Fragment):
            r += i.effective_key
        return r

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.param!r}, {self.types!r}, {self.inner!r})"
        )


@lru_cache
class Solutions(dict["_Param", list["Arguments | Fragment"]]):
    __slots__ = ("_com", "_gat", "_val")
    _com: "Arguments"  # a list during __init__
    _gat: "Arguments"  # a list during __init__

    def __call__(self, key: "_Param") -> tuple["Arguments", Value | None]:
        if su := self.get(key):  # could use .pop
            su = Arguments.from_list(su)
        else:
            su = self._gat if key.gatekeeper else self._com
        if va := self._val.get(key):
            Value.used.add(va)
        return su, va

    def __init__(self, args: "Arguments", ref: "ParaOMeta"):
        super().__init__()
        com: list[Arguments | Fragment]
        gat: list[Arguments | Fragment]
        val: dict[_Param, Value]
        self._com = com = []
        self._gat = gat = []
        self._val = val = {}
        div: bool = False  # com & gat diverged

        op = ref.__own_parameters__
        for arg in args:
            if isinstance(arg, Arguments):
                oth = Solutions(arg, ref)
                for k, v in oth._val.items():
                    val[k] = val.get(k) | v
                if len(ga := oth._gat) < len(co := oth._com):
                    div = True  # com & gat (will) divere
                # 1st
                for k, vs in self.items():
                    vs.extend(ga if k.gatekeeper else co)
                # 2nd
                for k, v in oth.items():
                    self[k].extend(v)
                # 3rd
                if co:
                    com.append(co)
                if ga:
                    gat.append(ga)
            elif (k := op.got(arg.param)) and arg.is_type_ok(ref):
                if arg.types:
                    com.append(arg)
                    gat.append(arg)
                    for vs in self.values():
                        vs.append(arg)
                if isinstance((v := arg.inner), Value):
                    Value.seen.add(v)
                    val[k] = val.get(k) | v
                else:
                    self[k].append(v)
            else:
                com.append(arg)
                if typd := bool(arg.types):
                    gat.append(arg)
                for k, vs in self.items():
                    if typd or not k.gatekeeper:
                        vs.append(arg)

        if self or val:
            self._com = Arguments.from_list(com)
        else:
            self._com = args
        if div or len(gat) < len(com):
            self._gat = Arguments.from_list(gat)
        else:
            self._gat = args

    def __missing__(self, key: "_Param") -> list["Arguments | Fragment"]:
        src = self._gat if key.gatekeeper else self._com
        self[key] = ret = src.copy()
        return ret


class Arguments(tuple["Arguments | Fragment", ...]):
    @classmethod
    def make(cls, *args: "Arguments | HasArguments | dict[KeyTE, Any]", **kwargs: Any):
        return cls._make(args, kwargs)

    @classmethod
    def _make(
        cls,
        args: "tuple[Arguments | HasArguments | dict[KeyTE, Any], ...]",
        kwargs: dict[str, Any] = None,
    ):
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
                    sub.extend(cls._from_dict(arg))
            else:
                raise TypeError(f"unsupported argument type: {type(arg)}")
        if kwargs:
            sub.extend(cls._from_dict(kwargs))

        return cls.from_list(sub)

    @classmethod
    def from_dict(
        cls,
        k2v: Mapping[KeyTE, Any] | Iterable[tuple[KeyTE, Any]],
        prio: PrioT = 0,
    ):
        if ret := cls(cls._from_dict(k2v, prio)):
            return ret
        return Arguments.EMPTY

    @classmethod
    def _from_dict(
        cls,
        k2v: Mapping[KeyTE, Any] | Iterable[tuple[KeyTE, Any]],
        prio: PrioT = 0,
    ):
        if callable(items := getattr(k2v, "items", None)):
            k2v = items()
        return (
            Fragment.make(k, v if isinstance(v, Value) else Value(v, prio))
            for k, v in k2v
        )

    @classmethod
    def from_list(cls, args: "list[Arguments | Fragment]") -> "Arguments":
        """Turn an iterable into arguments. Avoids unnecessary nesting or repeated creation of empty Arguments."""
        match args:
            case []:
                return cls.EMPTY
            case [Arguments()]:
                return args[0]
        return cls(args)

    def __repr__(self):
        return self.__class__.__name__ + (tuple.__repr__(self) if self else "()")

    def solve_value(self, param, owner, name) -> tuple["Arguments", "Value | None"]:
        return Solutions(self, owner)(param)

    @lru_cache
    def solve_class(
        self, ref: "ParaOMeta"
    ) -> "tuple[Arguments, Value[ParaOMeta] | None]":
        sub = []
        res = res0 = Value(ref, -inf)
        tar = {None, "__class__"}
        alt = False  # sub != self

        for arg in self:
            if isinstance(arg, Arguments):
                s, r = arg.solve_class(res.val)
                if s:
                    sub.append(s)
                    alt = alt or s is not arg
                if r is None:
                    continue
            elif (
                arg.param in tar
                and isinstance((r := arg.inner), Value)
                and arg.is_type_ok(res.val)
            ):
                Value.seen.add(r)
                if arg.param:
                    alt = True
                else:
                    sub.append(arg)
            else:
                sub.append(arg)
                continue
            res |= r
            if res.val is UNSET:
                res = res0

        if res is not res0:
            Value.used.add(res)
        return self.from_list(sub) if alt else self, None if res is res0 else res

    def get_root_of(self, seek: Value) -> Fragment | None:
        for root in self:
            curr = root
            while isinstance(curr, Fragment):
                curr = curr.inner
            if curr is seek:
                return root
            elif isinstance(curr, Arguments):
                if sub := curr.get_root_of(seek):
                    return sub

    def enumerate(
        self,
        used: bool | None = None,
        seen: bool | None = None,
        nested: bool = False,
    ):
        it = iter(self)
        done = {self}
        stack = [(it, tuple[Fragment, ...]())]

        while stack:
            it, path = stack[-1]
            for arg0 in it:
                arg = arg0
                while isinstance(arg, Fragment):
                    arg = arg.inner
                arg0: Fragment
                if isinstance(arg, Arguments):
                    if nested and arg not in done:
                        if arg is not arg0:
                            path += (arg0,)
                        it = iter(arg)
                        done.add(arg)
                        stack.append((it, path))
                        break
                else:
                    if (used is None or used == (arg in Value.used)) and (
                        seen is None or seen == (arg in Value.seen)
                    ):
                        yield path + ((arg,) if arg is arg0 else (arg0, arg))
            else:
                stack.pop()


Arguments.EMPTY = Arguments()
Arguments._ctxargs = ContextValue("ContextArguments", default=Arguments.EMPTY)


class HasArguments(Protocol):
    __args__: Arguments


eager = ContextValue[bool]("eager", default=False)


class OwnParameters(dict[str, "_Param"]):
    __slots__ = ("vals",)
    vals: set["_Param"]

    class CacheReset(RuntimeWarning): ...

    def __init__(self, cls: "ParaOMeta"):
        super().__init__(
            (name, param)
            for name in dir(cls)
            if not name.startswith("__")
            and isinstance((param := getattr(cls, name)), _Param)
        )
        self.vals = set(self.values())

    def got(self, key: "str | _Param") -> "_Param | None":
        if key in self.vals or (key := self.get(key)):
            return key

    @classmethod
    def reset(cls):
        if cls.cache:
            ewarn("partially filled __own_parameters__ cache reset", cls.CacheReset)
        cls.cache.clear()

    cache: dict["ParaOMeta", "OwnParameters"] = {}


class ParaOMeta(ABCMeta):
    @property
    def __fullname__(cls):
        return f"{cls.__module__}:{cls.__qualname__}"

    @property
    def __own_parameters__(cls) -> OwnParameters:
        if (val := OwnParameters.cache.get(cls)) is None:
            val = OwnParameters.cache[cls] = OwnParameters(cls)
        return val

    def __setattr__(cls, name, value):
        if not name.startswith("_"):
            OwnParameters.reset()
            if isinstance(value, _Param):
                value.__set_name__(cls, name)
        return super().__setattr__(name, value)

    def __delattr__(cls, name):
        if not name.startswith("_"):
            OwnParameters.reset()
            if isinstance((old := getattr(cls, name, None)), _Param):
                old.__set_name__(cls, None)
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
        arg = Arguments._make(args, kwargs)
        arg, val = arg.solve_class(cls)
        ret = cls.__new__(cls if val is None else val.val)
        ret.__args__ = arg
        ret.__init__()
        if eager():
            for name, param in ret.__class__.__own_parameters__.items():
                if param.eager:
                    getattr(ret, name)
        return ret


def get_inner_parao(value: Any):
    if isinstance(value, ParaO):
        yield value
    elif isinstance(value, Expansion):
        yield from value.expand()
    elif isinstance(value, (dict, list, tuple, set, frozenset)):
        queue = [iter((value,))]
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
                    yield curr
            else:
                queue.pop()


class ParaO(metaclass=ParaOMeta):
    __args__: Arguments  # | UNSET

    def __shash__(self, enc: _SHash) -> bytes:
        try:
            res = self.__shash
        except AttributeError:
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

    def __eq__(self, other):
        return (
            isinstance(self, ParaO)
            and self.__class__ is other.__class__
            and bin_hash(self) == bin_hash(other)
        )

    def __hash__(self) -> int:
        return int.from_bytes(bin_hash(self)[:8])

    @property
    def __inner__(self):
        for name, param in self.__class__.__own_parameters__.items():
            if param.significant:
                yield from get_inner_parao(getattr(self, name))

    def __repr__(
        self,
        *,
        compact: bool | str = False,
        param: "_Param" = None,
        param_name: str = None,
    ):
        if compact:
            if compact is True:
                compact = "..."
            items = [compact]
        else:
            items = [
                f"{name}={value!r}"
                for name, value, neutral in self.__rich_repr__()
                if (value is not UNSET if neutral is UNSET else value != neutral)
            ]
        ret = f"{self.__class__.__fullname__}({', '.join(items)})"
        if param is not None:
            if param_name is None:
                param_name = param._name(type(self)) or "???"
            ret += f".{param_name}={param!r}"
        return ret

    def __rich_repr__(self):
        for name, param in self.__class__.__own_parameters__.items():
            if param.significant and not isinstance(param, Const):
                if (neutral := param.neutral) is UNSET:
                    neutral = getattr(param, "default", UNSET)
                yield name, getattr(self, name), neutral


class UntypedWarning(RuntimeWarning):
    @classmethod
    def warn(cls, param: "_Param", instance: "ParaO", name: str | None = None):
        if name is None:
            name = param._name(type(instance))
        ewarn(f"{type(param)} {name} on {type(instance)}", category=cls)


class UntypedParameter(UntypedWarning): ...


type ExpansionFilter = (
    Collection[KeyE | Collection[KeyE]] | Callable[[Expansion, ParaO], bool]
)


class DuplicateParameter(RuntimeError): ...


@lru_cache
def _solve_name(param: "Param", icls: "ParaOMeta") -> str | None:
    lut = param._owner2name
    for cls in icls.__mro__:
        if cls in lut:
            return lut[cls]


@lru_cache
def _get_type_hints(cls: "ParaOMeta"):
    with catch_warnings(category=TypedAliasRedefined, action="ignore"):
        return get_type_hints(cls)


### actual code
class _Param[T]:
    significant: bool = True
    gatekeeper: bool = False
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

    def _cast(self, raw, typ):
        if typ is not UNSET and typ is not Any:
            try:
                exp = cast(raw, Expansion[typ])
            except TypeError:
                pass
            else:
                if isinstance(exp, Expansion):
                    raise exp
        return cast(raw, typ)

    def _get(self, val: Value | None, name: str, instance: "ParaO") -> T:
        raw = UNSET if val is None else val.val
        typ = self._type(type(instance), name)
        if typ is UNSET:
            UntypedParameter.warn(self, instance, name)
            return raw
        return self._cast(raw, typ)

    def _collect(self, expansion: "Expansion", instance: "ParaO"):
        return bool(self.collect) and (
            self.collect(expansion, instance)
            if callable(self.collect)
            else any(map(expansion.test, self.collect))
        )

    def _solve(
        self,
        val: Value | None,
        name: str,
        instance: "ParaO",
        *args: Arguments | Fragment,
    ):
        if val and val.prio < self.min_prio:
            val = None

        try:
            with Arguments._ctxargs(Arguments.from_list(args)):
                return self._get(val, name, instance)
        except Expansion as exp:
            exp.process(self, instance, val)
            exp.make = partial(self._solve, val, name, instance, *args)
            return exp
        except Exception as exc:
            exc.add_note(
                f"param stack: {instance.__repr__(compact=True, param=self, param_name=name)}"
            )
            raise

    @overload
    def __get__(self, instance: "ParaO", owner: type | None = None) -> T: ...
    @overload
    def __get__(self, instance: None | Any, owner: type | None = None) -> Self: ...

    def __get__(
        self, instance: "None | Any | ParaO", owner: type | None = None
    ) -> Self | T:
        if instance is None:
            return self
        cls = type(instance)
        name = self._name(cls)

        sub, val = instance.__args__.solve_value(self, cls, name)

        instance.__dict__[name] = raw = self._solve(val, name, instance, sub)
        return raw

    def __reduce__(self):
        for cls_name in self._owner2name.items():
            return getattr, cls_name
        raise PicklingError(f"Can't pickle a {type(self)} never used on a ParaO")

    min_prio: float = -inf
    eager: bool = True
    collect: ExpansionFilter = ()
    type: type | Unset
    type = UNSET


class Const[T](_Param[T]):
    def __init__(self, value: T, **kwargs):
        super().__init__(value=value, **kwargs)

    def __get__(self, instance, owner=None) -> T:
        return self if instance is None else self.value


class _DecoratorParam[T, F: Callable](_Param[T]):
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
        return super().__getattr__(name)

    func: F


class Prop[T](_DecoratorParam[T, Callable[[ParaO], T]]):
    def _type(self, cls, name):
        typ = getattr(self.func, "__annotations__", {}).get("return", UNSET)
        if typ is UNSET:
            typ = super()._type(cls, name)
        return typ

    def _get(self, val, name, instance) -> T:
        raw = super()._get(val, name, instance)
        return self.func(instance) if raw is UNSET else raw


class MissingParameterValue(TypeError): ...


class Param[T](_Param[T]):
    def __init__(self, default=UNSET, **kwargs):
        super().__init__(default=default, **kwargs)

    def _get(self, val, name, instance):
        raw = super()._get(val, name, instance)
        if raw is UNSET:
            if self.default is UNSET:
                raise MissingParameterValue(name)
            else:
                return self.default
        return raw


class ExpansionGeneratedKeyMissingParameter(RuntimeWarning): ...


class Expansion[T](BaseException):
    @classmethod
    def __cast_from__(cls, value, original_type):
        (typ,) = original_type.__args__
        result = cast(value, tuple[typ, ...])
        if isinstance(result, tuple):
            return cls(result)
        else:
            return NotImplemented

    def __init__(self, values: Iterable[T]):
        super().__init__()
        assert iter(values)  # ensure values is iterable
        self.values = values
        self._frames: list[tuple[ParaOMeta, Param, Value | None]] = []

    make: Callable[[Value], ParaO | Self] | None = None

    def test(self, item: KeyE | Iterable[KeyE]):
        match item:
            case _Param():
                return self.param is item
            case str():
                return bool(self.param_name == item)
            case type():
                return isinstance(self.source, item)
            case _:
                return all(map(self.test, item))

    def process(self, param: _Param, inst: ParaO, value: Value | None):
        # leave marks of origin
        if not self._frames:
            self.param = param
            self.source = inst
            # do we need these? or are they only for _unwind construction
            self.value = value
        # is it collected here?
        if param._collect(self, inst):
            return  # this will add ._get
        # keep track of key to dial-down to the origin
        self._frames.append(
            (inst.__class__, param, inst.__args__.get_root_of(self.value))
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
                    ewarn(
                        f"force added {rkey[0]!r}",
                        ExpansionGeneratedKeyMissingParameter,
                    )
                rkey.append(cls)
        return tuple(reversed(rkey))

    def expand(self, prio: PrioT = 0, **kwargs) -> Generator[ParaO, None, None]:
        key = self.make_key(**kwargs)
        for val in self.values:
            res = self.make(Fragment.make(key, Value(val, prio)))
            if isinstance(res, Expansion):
                yield from res.expand(prio=prio, **kwargs)
            else:
                yield res

    @staticmethod
    def generate(
        typ: ParaOMeta, args: Arguments, **kwargs
    ) -> Generator[ParaO, None, None]:
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
        parts = [f"<{safe_len(self.values, '???')} values>"]
        if self._frames:
            parts.append(f"from {self.source.__repr__(compact=True, param=self.param)}")
        return f"{self.__class__.__name__}({' '.join(parts)})"
