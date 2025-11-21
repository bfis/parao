from functools import partial
from itertools import chain, islice, takewhile
import json
import os
import pickle
from dataclasses import KW_ONLY, dataclass
from errno import EXDEV
from io import FileIO
from pathlib import Path
import re
from shutil import copy2, copytree, rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import GenericAlias, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Self,
    Sequence,
    Type,
    TypeAliasType,
    _AnnotatedAlias,
    overload,
)
from warnings import warn

from .action import RecursiveAction
from .core import (
    UNSET,
    AbstractParam,
    Const,
    ParaO,
    Param,
    Unset,
    UntypedWarning,
    get_inner_parao,
)
from .print import PPrint
from .run import _Output, _RunAction, _Template, PseudoOutput, _RunAct
from .shash import hex_hash, primitives

type JSON[T] = T
type Pickle[T] = T


class FSOutput(Path, PseudoOutput):
    tmpio: None | FileIO = None
    _temp: None | TemporaryDirectory = None

    def __init__(self, path, *paths):
        super().__init__(path, *paths)
        self._copy_temp(path)

    def _copy_temp(self, source):
        if isinstance(source, FSOutput):
            tmp = source._temp
            if isinstance(tmp, TemporaryDirectory) and self.is_relative_to(tmp.name):
                self._temp = tmp
        return self

    def with_segments(self, *args):
        return super().with_segments(*args)._copy_temp(self)

    @classmethod
    def tempDir(cls, **kwargs):
        tmp = TemporaryDirectory(**kwargs)
        ret = cls(tmp.name)
        ret._temp = tmp
        return ret

    def close(self):
        if (f := self.tmpio) is not None and not f.closed:
            f.close()


class Dir(FSOutput):
    @classmethod
    def temp(cls, **kwargs):
        return cls.tempDir(**kwargs)


class File(FSOutput):
    @classmethod
    def temp(cls, mode: str = "wb", **kwargs):
        tmp = NamedTemporaryFile(mode, **kwargs)
        ret = cls(tmp.name)
        ret.tmpio = tmp.file
        ret._closer = tmp._closer
        return ret


class UntypedOuput(UntypedWarning): ...


class MoveAcrossFilesystem(RuntimeWarning): ...


class MissingOuput(FileNotFoundError): ...


class NotSupported(TypeError, NotImplementedError): ...


class Incompatible(RuntimeError): ...


class Inconsistent(RuntimeWarning): ...


@dataclass
class Coder[T]:
    """En-/De-coder definition, including suffix."""

    suffix: str
    tat: TypeAliasType | None = None
    load: None | Callable[[], T] = None
    dump: None | Callable[[T], None] = None
    _: KW_ONLY
    typ: type | None = None
    text: bool = False

    @property
    def is_dir(self):
        return self.typ is not None and issubclass(self.typ, Dir)

    def match(self, hint: TypeAliasType | type):
        if isinstance(hint, TypeAliasType):
            return hint == self.tat
        elif self.typ is not None:
            if hint is None:
                return isinstance(hint, self.typ)
            return isinstance(hint, type) and issubclass(hint, self.typ)

    def conform(self, data: FSOutput) -> T:
        if (have := self.is_dir) != (want := data.is_dir()):
            have = "directory" if have else "file"
            want = "directory" if want else "file"
            exc = IsADirectoryError if have else NotADirectoryError
            raise exc(f"got {have} expected a {want}")
        if (ftyp := self.typ) is None:
            ftyp = FSOutput
        if not isinstance(data, ftyp):
            warn(f"got a {type(data)}, expected a {ftyp}", Inconsistent)
            data = ftyp(data)
        return data

    @property
    def bt_mode(self):
        return "t" if self.text else "b"


# further (_)RunAct(ion) subclass, to enable type hints
class RunAct[R, A: RunAction, I: Task](_RunAct[R, A, I]):
    output: "Output[R, Self]"


class RunAction[R](_RunAction[R]):
    output_template_attribute_name = "output"
    _act = RunAct

    if TYPE_CHECKING:

        @overload
        def __get__[I: ParaO](
            self, inst: I, owner: type | None = None
        ) -> RunAct[R, Self, I]: ...
        @overload
        def __get__(self, inst: None | RunAct, owner: type | None = None) -> Self: ...


class Output[R, A: RunAct](_Output[R, A]):
    """
    Basic output implementation for local file storage.
        Uses pickle by default, but also supports
        JSON and direct File and Dir output.
    """

    __slots__ = ("coder", "path", "tmp_dir")
    coder: Coder[R]
    tmp_dir: Path
    path: Path

    # temp file/dir utility
    @overload
    def temp(
        self: "Output[TypeAliasType]", mode: str | None = None, **kwargs
    ) -> File: ...

    @overload
    def temp[F: FSOutput](self: "Output[F]") -> F: ...

    @overload
    def temp(self, mode: str | None = None, **kwargs) -> File: ...

    def temp(self, mode: str | None = None, **kwargs) -> FSOutput:
        path = self.path
        dps = dict(
            dir=self.tmp_dir,
            prefix=path.with_suffix(".tmp").name,
            suffix=path.suffix,
        )
        dps["dir"].mkdir(parents=True, exist_ok=True)

        if (is_dir := self.coder.is_dir) or mode == "":
            if mode or kwargs:
                raise ValueError("superfluous arguments!")
            ret = self.fsoutput_type.tempDir(**dps)
            return ret if is_dir else ret.joinpath(path.with_stem("temp").name)

        if mode is None:
            mode = "w" + self.coder.bt_mode

        return self.fsoutput_type.temp(mode, **dps, **kwargs)

    # implement the abstracts
    @property
    def exists(self):
        return self.path.exists()

    def remove(self, missing_ok: bool = False):
        if self.coder.is_dir:
            if not missing_ok or self.exists:
                rmtree(self.path)
                # self.path.rmdir(missing_ok)
        else:
            self.path.unlink(missing_ok)

    def load(self) -> R:
        if callable(load := self.coder.load):
            with self.path.open("rb") as f:
                return load(f)
        else:
            return self.type(self.path)

    def _temp_copy(self, data: Path) -> FSOutput:
        if self.coder.is_dir:
            return copytree(data, self.temp(), dirs_exist_ok=True)
        else:
            return copy2(data, self.temp(""))

    def _dump(self, data: FSOutput) -> FSOutput:
        if not data.exists():
            raise MissingOuput(str(data))
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        data: FSOutput = self.coder.conform(data)
        data.close()
        if not (data._temp or data.tmpio):  # dont steal non-temporaries
            data = self._temp_copy(data)
            # no need to attempt EXDEV recovery via another _temp_copy
            diff_dev = False
        elif diff_dev := (data.stat().st_dev != path.parent.stat().st_dev):
            warn("may be slow or fail", MoveAcrossFilesystem, stacklevel=3)
        try:
            data = data.replace(path)
        except OSError as err:
            if err.errno == EXDEV and diff_dev:
                warn("failed, making explicit copy", MoveAcrossFilesystem, stacklevel=3)
                # try again
                data = self._temp_copy(data)
                data = data.replace(path)
            else:
                raise
        return data

    def dump(self, data: R | FSOutput) -> R:
        if isinstance(data, FSOutput):
            data = self._dump(data)
            if (
                isinstance(self.type, type)
                and issubclass(self.type, FSOutput)
                and isinstance(data, self.type)
            ):
                return data
            else:
                return self.load()
        elif callable(dump := self.coder.dump):
            if isinstance(data, PseudoOutput):
                raise NotSupported(f"wont handle {type(data)}")
            tmp = self.temp("w" + self.coder.bt_mode)
            dump(data, tmp.tmpio)
            self._dump(tmp)
            return data
        else:
            raise NotSupported(f"got {type(data)}, expected {self.type}")

    # type stuff
    @property
    def fsoutput_type(self) -> Type[FSOutput]:
        for typ in self.unpacked_types:
            if isinstance(typ, type) and issubclass(typ, FSOutput):
                return typ
        return Dir if self.coder.is_dir else File

    @property
    def unpacked_types(self) -> Iterable[type]:
        """unwrap type(s) to get a isinstance-able base"""
        queue = [self.type]
        while queue:
            typ = queue.pop()
            if typ is UNSET:
                continue
            while isinstance(typ, (GenericAlias, _AnnotatedAlias)):
                typ = typ.__origin__
            if isinstance(typ, UnionType):
                queue.extend(typ.__args__)
            else:
                yield typ

    @property
    def type(self) -> type | Unset:  # this bricks "type" in here
        action = self.act.action
        return getattr(action.func, "__annotations__", {}).get(
            "return", getattr(action, "return_type", UNSET)
        )


class PlainTemplate(_Template):
    def __call__(self, act: RunAct):
        out = self._output_cls(act)
        out.coder = self._coder(out)
        out.path = self._path(out)
        out.tmp_dir = self._tmp_dir(out)
        return out

    def _coder(self, out: Output) -> Coder:
        typs = tuple(out.unpacked_types)
        if not typs:
            UntypedOuput.warn(out.act.action, out.act.instance)
            return self._coders[-1]
        for enc in chain(self.coders_extra, self._coders):
            for typ in typs:
                if enc.match(typ):
                    return enc
        raise NotSupported(f"no coder for {typ}")

    def _name(self, act: RunAct) -> str:
        return f"{act.name}_{hex_hash(act.instance)}"

    def _dirs(self, inst: ParaO) -> Sequence[str]:
        return ()

    def _dirs_filter(self, dirs: Iterable[str | None]) -> Iterable[str]:
        return filter(None, dirs)

    def _path(self, out: Output) -> Path:
        return Path(
            self.dir_base,
            *self._dirs_filter(self._dirs(out.act.instance)),
            self._name(out.act),
        ).with_suffix(out.coder.suffix)

    def _tmp_dir(self, out: Output) -> Path:
        if tmp_dir := self.dir_temp:
            return Path(tmp_dir)
        else:
            return out.path.parent

    dir_base = Param[str]()
    dir_temp = Param[str | None](None)
    coders_extra = Param[list[Coder] | tuple[Coder, ...]](())

    _output_cls: Type[Output] = Output
    _coders = (
        Coder(".dir", typ=Dir),
        Coder(".file", typ=File),
        Coder(".json", JSON, json.load, json.dump, text=True),
        Coder(".pkl", Pickle, pickle.load, pickle.dump, typ=object),
    )


class FancyTemplate(PlainTemplate):
    module_name = Param[bool](True)
    class_name = Param[bool](True)

    bad_chars = Param[str](os.sep)
    bad_replacer = Param[str]("_")

    name_limit = Param[int](-255)
    name_ellipsis = Param[str]("…")

    dir_limit = Param[int](0)
    tot_limit = Param[int](-4000)
    # only used if the actual limit was programmatically determined:
    tot_reserve = Param[int](50)

    def _typ_name(self, typ: type):
        ret: list[str] = []
        if self.module_name and (m := typ.__module__):
            ret.append(m)
        if self.class_name and (q := typ.__qualname__):
            ret.append(q)
        return ":".join(ret)

    def _dirs(self, inst: ParaO):
        yield from super()._dirs(inst)
        yield self._typ_name(inst.__class__)
        yield from self._encode_parao(inst, set())

    @property
    def _fix_bad(self):
        return partial(
            re.compile(f"[{re.escape(self.bad_chars)}]+").sub, self.bad_replacer
        )

    _fix_bad: Callable[[str], str]

    @property
    def _fix_len(self):
        ellipsis = self.name_ellipsis

        encode = os.fsencode
        limit = self.name_limit
        if limit < 0:
            try:
                # limit = os.statvfs(self.dir_base).f_namemax
                limit = os.pathconf(self.dir_base, "PC_NAME_MAX")
            except AttributeError:
                limit = -limit
        target = limit - len(encode(ellipsis))
        assert target > 0

        def fix(data: str) -> str:
            if 0 < limit < len(encode(data)):
                data = data[:target]
                while target < len(encode(data)):
                    data = data[:-1]
                data += ellipsis
            return data

        return fix

    _fix_len: Callable[[str], str]

    def _dirs_filter(self, dirs):
        if (max_dir := self.dir_limit) < 0:
            return ()

        dirs = super()._dirs_filter(dirs)
        dirs = map(self._fix_bad, dirs)
        dirs = map(self._fix_len, dirs)

        if max_dir > 0:
            dirs = islice(dirs, max_dir)

        if max_tot := self.tot_limit:
            if max_tot < 0:
                try:
                    max_tot = os.pathconf(self.dir_base, "PC_PATH_MAX")
                except (AttributeError, ValueError):
                    max_tot = -max_tot
                else:
                    max_tot -= self.tot_reserve

            tot = 0

            def cond(name: str) -> bool:
                nonlocal tot
                tot += len(name) + 1  # i.e. len(os.sep)
                return tot < max_tot

            dirs = takewhile(cond, dirs)

        return dirs

    small_join = Param[str]("_")
    label_patt = Param[str]("{0}={1}")
    small_mod = Param[int](1)
    label_mod = Param[int](100)
    default_pos = Param[float | None](None)

    def _encode_parao(self, inst: ParaO, seen: set[ParaO]):
        seen.add(inst)

        cand: list[tuple[float, str, str | Iterable[ParaO]]] = []
        rest: list[Iterable[ParaO]] = []

        for name, param in inst.__class__.__own_parameters__.items():
            if not param.significant:
                continue

            val = getattr(inst, name)
            nut = param.neutral
            if nut is not UNSET and (val is nut or val == nut):
                continue

            if (pos := getattr(param, "pos", self.default_pos)) is None:
                rest.append(get_inner_parao(val))
            else:
                if (enc := self._encode_value(val)) is None:
                    enc = get_inner_parao(val)
                cand.append((pos, name, enc))

        cand.sort()

        smod = self.small_mod
        lmod = self.label_mod

        small = _Small(self.small_join.join)
        for pos, name, enc in cand:
            if isinstance(enc, str):
                if not smod or pos % smod:  # want small
                    small.append(enc)
                else:
                    if small:
                        yield small.flush()
                    if not (lmod and pos % lmod):  # want label
                        enc = self.label_patt.format(name, enc)
                    yield enc
            else:
                if small:
                    yield small.flush()
                for sub in enc:
                    if sub in seen:
                        continue
                    tmpl: FancyTemplate = getattr(self, "output", self)
                    yield from tmpl._encode_parao(sub, seen)
        if small:
            yield small.flush()

        for res in rest:
            for sub in res:
                if sub in seen:
                    continue
                tmpl: FancyTemplate = getattr(self, "output", self)
                yield from tmpl._encode_parao(sub, seen)

    sep_pair = Param[str | list[str] | tuple[str, ...]](":")
    sep_item = Param[str | list[str] | tuple[str, ...]](",")

    def _encode_value(self, raw: Any) -> str | None:
        try:
            return self._encode_step(self.sep_pair, self.sep_item, raw)
        except (
            IndexError,  # structure too deep, ran out of seperators
            NotImplementedError,  # unsupported type
        ):
            return None

    def _encode_step(self, pair: Sequence[str], item: Sequence[str], raw: Any) -> str:
        if isinstance(raw, bytes):
            return raw.hex()
        if isinstance(raw, primitives):
            return str(raw)

        joiner = item[0].join  # trigger IndexError early
        if isinstance(raw, dict):
            pjoin = pair[0].join  # trigger IndexError early
            pfunc = partial(self._encode_step, pair[1:], item[1:])
            return joiner(pjoin((pfunc(k), pfunc(v))) for k, v in raw.items())
        elif isinstance(raw, (tuple, list, set, frozenset)):
            vals = map(partial(self._encode_step, pair, item[1:]), raw)
            if isinstance(raw, (set, frozenset)):
                vals = sorted(vals)
            return joiner(vals)

        raise NotImplementedError


class _Small(list[str]):
    __slots__ = ("joiner",)

    def __init__(self, joiner: Callable[[Self], str]):
        super().__init__()
        self.joiner = joiner

    def flush(self):
        ret = self.joiner(self)
        self.clear()
        return ret


# need them here, otherwise we get cyclic imports with task.py
pprint = PPrint()


class Task[R](ParaO):
    code_version: Const
    run: RunAction[R]
    output = Param[FancyTemplate](significant=False)

    def __init_subclass__(cls):
        v = cls.__dict__.get("code_version")
        if v is not None and not isinstance(v, AbstractParam):
            cls.code_version = Const(v)
        r = cls.__dict__.get("run")
        if r is not None and not isinstance(r, AbstractParam):
            cls.run = RunAction(r)
        return super().__init_subclass__()

    @RecursiveAction
    def remove(self, depth: int):
        out = self.run.output
        if out.exists:
            out.remove()
            after = "removed"
        else:
            after = "missing"
        pprint.pprint(self, indent=2 * depth, after=after)

    @RecursiveAction
    def status(self, depth: int):
        after = "done" if self.run.done else "missing"
        pprint.pprint(self, indent=2 * depth, after=after)

    @RecursiveAction
    def print(self, depth: int):
        pprint.pprint(self, indent=2 * depth)
