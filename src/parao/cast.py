from itertools import repeat
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

        # container types
        if isinstance(ori, type):  # pragma: no branch
            if isinstance(val, (str, bytes)):
                raise CastError(type(val), ori)

            if issubclass(ori, tuple):
                if not args:
                    if val:
                        raise TypeError("too many values given")
                    return val if isinstance(val, ori) else ori()
                if args[1:] == (Ellipsis,):
                    return ori(map(cast, val, repeat(args[0])))

                ret = ori(map(cast, val, args))
                if len(ret) != len(args):
                    raise TypeError("wrong number of values")
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

        raise TypeError(f"type no understood: {typ}")
    elif typ is Any:
        return val
    # primitive types
    elif typ is None or typ is NoneType:
        if val is not None:
            raise TypeError("invalid value for None")
        return None
    elif isinstance(val, typ):
        return val
    elif isinstance(val, str) and (
        typ is bool or isinstance(type, typ) and issubclass(typ, bool)
    ):
        raise TypeError(f"can't cast {val!r} to {typ}")
    else:
        ret = typ(val)
        if isinstance(val, _numeric) and isinstance(ret, _numeric) and ret != val:
            raise ValueError(f"cast to {typ} not accurate: {val!r}!={ret!r}")
        return ret
