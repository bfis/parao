from itertools import repeat
from types import UnionType, NoneType
from typing import _AnnotatedAlias, Any, Protocol, Union, get_args, get_origin

_numeric = int, float, complex


class Opaque:
    """Ignored during casting unless they are Castable."""


class Castable(Protocol):
    @classmethod
    def __cast_from__(cls, value, original_type): ...


def cast(val: Any, typ: type) -> Any:
    typ0 = typ  # the orignal type
    if isinstance(typ, _AnnotatedAlias):
        (typ,) = typ.__args__
    ori = get_origin(typ)

    if cast_to := getattr(val, "__cast_to__", None):
        try:
            return cast_to(typ, typ0)
        except NotImplementedError:
            pass

    if cast_from := getattr(ori or typ, "__cast_from__", None):
        try:
            return cast_from(val, typ0)
        except NotImplementedError:
            pass

    if isinstance(val, Opaque):
        return val

    if ori:
        args = get_args(typ)

        if ori is UnionType or ori is Union:
            for arg in args:
                try:
                    return cast(val, arg)
                except (TypeError, ValueError):
                    pass
            else:
                raise TypeError(f"could not cast {val!r} to {typ}")

        # container types
        if isinstance(ori, type):  # pragma: no branch

            if isinstance(val, (str, bytes)):
                raise ValueError(f"can't cast {type(val)} to container")

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
                typK, typV = args
                return ori(
                    {
                        cast(k, typK): cast(v, typV)
                        for k, v in (val.items() if isinstance(val, dict) else val)
                    }
                )

        raise TypeError(f"type no understood: {typ}")
    elif typ is Any:
        return val
    # primitive types
    elif typ is None or typ is NoneType:
        if val is not None:
            raise ValueError("invalid value for None")
        return None
    elif isinstance(val, typ):
        return val
    else:
        ret = typ(val)
        if isinstance(val, _numeric) and isinstance(ret, _numeric) and ret != val:
            raise ValueError(f"cast to {typ} not accurate: {val!r}!={ret!r}")
        return ret
