from collections import defaultdict
from .core import AbstractParam, Fragment, KeyTE, ParaO, get_inner_parao
from .misc import ewarn


def _match_recusive(instance: ParaO, fragments: tuple[Fragment]):
    cls = instance.__class__
    op = cls.__own_parameters__

    com: list[Fragment] = []
    sub: defaultdict[AbstractParam, list[Fragment]] = defaultdict(com.copy)
    for frag in fragments:
        com.append(frag)
        for s in sub.values():
            s.append(frag)

        if (param := op.got(frag.param)) and frag.is_type_ok(cls):
            if frag.inner is None:
                yield instance, param
            else:
                sub[param].append(frag.inner)

    for param in op.values():
        frags = tuple(sub[param]) if param in sub else fragments
        for inner in get_inner_parao(param.__get__(instance)):
            yield from _match_recusive(inner, frags)


def find_value(self: ParaO, key: KeyTE):
    frag = Fragment.make(key, None)
    cand = _match_recusive(self, (frag,))

    try:
        inst0, param0 = next(cand)
    except StopIteration:
        raise NoSuchParameter(str(key))

    first = True
    for inst, param in cand:
        if param is param0 and param.significant and inst == inst0:
            continue
        if first:
            value0 = param0.__get__(inst0)
        if param.__get__(inst) == value0:
            if first:
                ewarn(str(key), MultipleSameValues, stacklevel=2)
                first = False
        else:
            raise InconsistentValues(str(key))
    if first:
        value0 = param0.__get__(inst0)

    return value0


class NoSuchParameter(AttributeError): ...


class InconsistentValues(ValueError): ...


class MultipleSameValues(RuntimeWarning): ...


class ItemSteno(ParaO):
    __getitem__ = find_value


class AttrSteno(ParaO):
    def __getattr__(self, name: str):
        if not name.startswith("_"):
            return find_value(self, name)
        return super().__getattr__(name)


class Steno(ItemSteno, AttrSteno): ...
