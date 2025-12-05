from collections.abc import Callable
from functools import lru_cache
from inspect import signature
from itertools import repeat
from numbers import Number
from types import NoneType, UnionType
from typing import Any, Protocol, Union, _AnnotatedAlias, get_args, get_origin

_numeric = int, float, complex


class CastError(TypeError, ValueError):
    def __init__(self, src, dst):
        super().__init__(f"{src!r} -> {dst}")

    def __str__(self):
        return f"cast failed: {super().__str__()}"


class Opaque:
    """Ignored during casting unless they are Castable."""


class Castable(Protocol):
    @classmethod
    def __cast_from__(cls, value, original_type): ...


@lru_cache
def sigcheck(func: Callable, args: tuple, ret) -> bool:
    if args is ...:
        return True
    try:
        signature(func).bind(*args)
    except TypeError:
        return False
    return True


def cast(val: Any, typ: type) -> Any:
    typ0 = typ  # the orignal type
    if isinstance(typ, _AnnotatedAlias):
        typ = typ.__origin__
    ori = get_origin(typ)

    if cast_to := getattr(val, "__cast_to__", None):
        if (ret := cast_to(ori or typ, typ0)) is not NotImplemented:
            return ret

    if cast_from := getattr(ori or typ, "__cast_from__", None):
        if (ret := cast_from(val, typ0)) is not NotImplemented:
            return ret

    if isinstance(val, Opaque):
        return val

    if ori:
        args = get_args(typ)

        if ori is UnionType or ori is Union:
            err = CastError(val, typ)
            for arg in args:
                try:
                    return cast(val, arg)
                except (TypeError, ValueError) as exc:
                    err.add_note(str(exc))
            raise err

        if ori is Callable:
            if callable(val) and (
                not args
                or sigcheck(
                    val,
                    tuple(args[0]) if isinstance(args[0], list) else args[0],
                    args[1],
                )
            ):
                return val
            raise CastError(val, typ)

        # container types
        if isinstance(ori, type):  # pragma: no branch
            if isinstance(val, (str, bytes)):
                raise CastError(type(val), ori)

            if issubclass(ori, tuple):
                if not args:
                    if val:
                        raise ValueError("too many values given")
                    return val if isinstance(val, ori) else ori()
                if args[1:] == (Ellipsis,):
                    return ori(map(cast, val, repeat(args[0])))

                ret = ori(map(cast, val, args))
                if len(ret) != len(args):
                    raise ValueError("wrong number of values")
                return ret

            if issubclass(ori, (list, set, frozenset)):
                (typ1,) = args
                return ori(map(cast, val, repeat(typ1)))

            if issubclass(ori, dict):
                k_typ, v_typ = args
                return ori(
                    {
                        cast(k, k_typ): cast(v, v_typ)
                        for k, v in (val.items() if isinstance(val, dict) else val)
                    }
                )

    elif typ is Any:
        return val
    # primitive types
    elif typ is None or typ is NoneType:
        if val is not None:
            raise CastError(val, None)
        return None
    elif isinstance(typ, type):
        if isinstance(val, typ):
            return val
        elif isinstance(val, Number) and issubclass(typ, Number):
            ret = typ(val)
            if not isinstance(ret, bool) and ret != val:
                raise ValueError(f"cast to {typ} not accurate: {val!r}!={ret!r}")
            return ret
        elif (
            issubclass(typ, bool)
            or (isinstance(val, int) and issubclass(typ, (bytes, bytearray)))
            or (
                issubclass(typ, str)
                and isinstance(val, (tuple, list, dict, set, frozenset))
            )
        ):
            raise CastError(val, typ)
        else:
            return typ(val)

    raise TypeError(f"type no understood: {typ}")
