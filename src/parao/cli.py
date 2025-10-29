import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from operator import attrgetter
from typing import get_origin
from warnings import warn

from .action import Plan
from .cast import cast
from .core import AbstractParam, Arg, Arguments, Expansion, ParaO, ParaOMeta, eager
from .misc import PeekableIter, is_subseq


class Value(str):
    __slots__ = ("empty", "position")

    def __new__(cls, value: str, position: int):
        self = super().__new__(cls, "" if value is None else value)
        self.empty = value is None
        self.position = position
        return self

    def __cast_to__(self, typ, original_type):
        if ori := get_origin(typ):
            if isinstance(ori, type):  # pragma: no branch
                if issubclass(ori, (tuple, list, set, frozenset)):
                    if len(parts := self.split(",")) > 1:
                        return cast(parts, original_type)
        if self.empty and typ is bool:
            return True

        raise NotImplementedError


class MalformedCommandline(ValueError): ...


class ParaONotFound(LookupError): ...


class NotAParaO(ValueError): ...


class Sep(tuple[str, ...]):
    class NeedValues(RuntimeError): ...

    class Overlap(RuntimeError): ...

    def __init__(self, _):
        super().__init__()
        if not self:
            raise self.NeedValues()

    def __lshift__(self, other: "Sep") -> "Sep":
        if overlap := set(self).intersection(other):
            raise self.Overlap(f"{self} & {other} = {overlap}")
        return self.__class__(self + other)

    def __or__(self, other: "Sep") -> "Sep":
        return self.__class__(set(self).union(other))

    @cached_property
    def parts(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (
            tuple(e for e in self if len(e) == 1),
            tuple(e for e in self if len(e) > 1),
        )

    @cached_property
    def regex(self):
        char, frag = self.parts
        if frag:
            frag = tuple(map(re.escape, frag))
        if char:
            frag += (f"[{re.escape("".join(char))}]",)
        return re.compile(f"(?:{"|".join(frag)})" if len(frag) > 1 else frag[0])

    def split1(self, string: str) -> tuple[str, str | None]:
        res = self.regex.split(string, 1)
        if len(res) > 1:
            return tuple(res)
        else:
            return res[0], None

    def split(self, string: str) -> list[str]:
        return self.regex.split(string)

    def sub(self, repl: str, string: str) -> str:
        return self.regex.sub(repl, string)


@dataclass
class CLIParser:
    uscore: Sep = "-"
    chain: Sep = "."
    module: Sep = ":"
    part: Sep = ",+"

    pair: Sep = ":"
    item: Sep = ","

    flag: Sep = ";"
    value: Sep = "="

    def __post_init__(self):
        for key, typ in CLIParser.__annotations__.items():
            if typ is Sep:  # pragma: no branch
                setattr(self, key, Sep(getattr(self, key)))

        outer = self.flag | self.value
        self._flag_value_disjoint = set(self.flag).isdisjoint(self.value)

        self.uscore << self.chain << self.module << self.part << outer
        self.pair << self.item << outer

    def argument(self, raw: str):
        if self._flag_value_disjoint:
            raw, value = self.value.split1(raw)
            raw, flags = self.flag.split1(raw)
        else:
            raw, flags = self.flag.split1(raw)
            if flags is None:
                value = None
            else:
                flags, value = self.value.split1(flags)

        if flags is not None:
            flags = dict(map(self.pair.split1, self.item.split(flags)))

        key = list(
            filter(
                any,
                map(
                    self.element,
                    self.part.split(self.uscore.sub("_", raw)),
                ),
            )
        )

        return key, flags, value

    def element(self, raw: str):
        mod, att = self.module.split1(raw)
        return (
            self.chain.sub(".", mod),
            self.chain.sub(".", att) if att else att,
        )


class NoAttributeBoundary(RuntimeWarning): ...


class UnsupportedKeyType(RuntimeWarning): ...


class MultipleCandidates(RuntimeWarning): ...


class AmbigouusCandidate(RuntimeWarning): ...


class UnusedCLIArguments(RuntimeWarning): ...


class CLI:
    prase_raw = CLIParser()

    def __init__(self):

        seen = set()
        queue: list[type] = [ParaO]
        for curr in queue:
            queue.extend(
                cand
                for cand in reversed(curr.__subclasses__())
                if cand.__name__[0] != "_"
                and (cand.__module__[0] != "_" or cand.__module__ == "__main__")
                and cand not in seen
            )
            seen.add(curr)

        self._paraos = seen

    @cached_property
    def find_parao(self):
        lut = defaultdict(dict)
        for s in self._paraos:
            qn = s.__qualname__.split(".")
            for i in range(len(qn)):
                sub = lut[".".join(qn[i:])]
                k = tuple(s.__module__.split("."))
                sub[k] = False if k in sub else s

        def func(module: str, attr: str) -> ParaOMeta | None:
            if sub := lut.get(attr, None):
                if module:
                    want = module.split(".")
                    cand = []

                    for have, parao in sub.items():
                        if is_subseq(want, have):
                            cand.append(parao)
                else:
                    cand = sub.values()

                if num := len(cand):
                    if num > 1:
                        w = MultipleCandidates
                    else:
                        if ret := next(iter(cand)):
                            return ret
                        else:
                            w = AmbigouusCandidate
                    warn(f"{module}:{attr}" if module else attr, w)

        return func

    def _split_case(self, raw: str):
        parts = raw.split(".")
        if len(parts) == 1:
            if raw[0].isupper():
                return "", raw, ""
        else:
            upper = [p[0].isupper() for p in parts]
            try:
                b = upper.index(True)
            except ValueError:
                pass
            else:
                upper.append(False)  # easiert than handling the -1
                e = upper.index(False, b)
                return ".".join(parts[:b]), ".".join(parts[b:e]), ".".join(parts[e:])
        return "", "", raw

    def _parse_mod_att(
        self,
        mod_att: tuple[str, str | None],
        typ: type | tuple[type],
        typ_bad: type[Warning | Exception],
    ):
        module, attr = mod_att

        if attr is None:
            module, cname, sub = self._split_case(module)
            # prepare attribute for import lookup
            attr = ".".join(filter(None, (cname, sub)))
        elif attr:
            pre, cname, sub = self._split_case(attr)
            if pre:  # some non-module prefix
                cname = ""  # skip lookup by subclass
        if attr and cname:  # lookup by subclass
            if ret := self.find_parao(module, cname):
                return attrgetter(sub)(ret) if sub else ret

        if module:
            if not attr:
                raise MalformedCommandline(f"Missing attribute for module {module}:")
            try:
                ret = attrgetter(attr)(import_module(module))
            except (ModuleNotFoundError, AttributeError) as e:
                e.add_note(f"module: {module}, attribute: {attr}")
                raise

            if not isinstance(ret, typ):  # pragma: no branch
                if issubclass(typ_bad, Warning):
                    warn(repr(ret), typ_bad)
                else:
                    raise typ_bad(repr(ret))

            return ret

        return module

    def parse_typ(self, raw: str):
        return self._parse_mod_att(self.prase_raw.element(raw), ParaOMeta, NotAParaO)

    def parse_key(self, mod_att):
        return (
            self._parse_mod_att(
                mod_att,
                (type, AbstractParam, str)[:2],  # no strings
                UnsupportedKeyType,
            )
            or mod_att[0]
        )

    def parse_args(self, args: list[str]):
        pre: list[str] = []
        got: list[tuple[ParaOMeta, Arguments, list[str]]] = []

        typ: type = None
        raw: list[str] = None
        curr: list[Arg] = None

        pit = PeekableIter(args)
        for arg in pit:
            if not arg:  # ignore empty standalone args
                continue
            if body := arg.lstrip("+-"):
                if start := arg[: -len(body)]:
                    if typ is None:
                        pre.append(arg)
                        continue

                    key, flags, value = self.prase_raw.argument(body)

                    key = tuple(map(self.parse_key, key))

                    if flags is None:
                        flags = {}

                    # fill value if it makes sense
                    if value is None:
                        if not pit.peek("+").startswith(("+", "-")):
                            value = next(pit)
                    if "json" in flags and value:
                        try:
                            value = json.loads(value)
                        except Exception as e:
                            e.add_note(f"for argument {arg}")
                            raise
                    else:
                        value = Value(value, len(curr))

                    # prio
                    if prio := flags.get("prio", None):
                        try:
                            prio = int(prio)
                        except ValueError:
                            try:
                                prio = float(prio)
                            except ValueError as e:
                                e.add_note(f"for argument: {arg}")
                                raise
                    else:
                        prio = 1 - start.count("-") + start.count("+")

                    raw.append(arg)
                    curr.append(Arg(key, value, prio))
                else:
                    if typ is not None:
                        got.append((typ, Arguments(curr), raw))
                    # solve typ
                    typ = self.parse_typ(body)
                    if not typ:
                        raise ParaONotFound(body)
                    raw = [arg]
                    curr = []
            else:
                break

        if typ is not None:
            got.append((typ, Arguments(curr), raw))

        return pre, got, list(pit)

    def _wrap(self, pre: list[str], post: list[str]):
        if pre:
            warn(f"{pre=}", UnusedCLIArguments)
        if post:
            warn(f"{post=}", UnusedCLIArguments)

        return nullcontext(self._consume)

    def _consume(self, typ: ParaOMeta, args: Arguments, raw: list[str] = ()):
        try:
            yield from Expansion.generate(typ, args)
        except Exception as exc:
            exc.add_note(f"for arguments: {" ".join(raw)}")
            raise

    def _run(self, args: list[str]):
        pre, got, post = self.parse_args(args)

        with eager(True), self._wrap(pre, post) as consume:
            for item in got:
                yield from consume(*item)

    def run(self, args: list[str] | None = None, **kwargs):
        if args is None:
            args = sys.argv[1:]

        kwargs.setdefault("run", True)

        with Plan().use(**kwargs):
            return list(self._run(args))


if __name__ == "__main__":
    CLI().run()  # pragma: no cover
