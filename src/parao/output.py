import json
import pickle
from dataclasses import KW_ONLY, dataclass
from io import FileIO
from pathlib import Path
from shutil import copy2, copytree, rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import GenericAlias, UnionType
from typing import Callable, Iterable, Type, TypeAliasType, _AnnotatedAlias, overload
from warnings import warn

from .core import UNSET, Unset, UntypedWarning
from .run import BaseOutput, PseudoOutput
from .shash import hex_hash

type JSON[T] = T
type Pickle[T] = T


class FSOutput(Path, PseudoOutput):
    file: None | FileIO = None
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


class Dir(FSOutput):
    @classmethod
    def temp(cls, **kwargs):
        return cls.tempDir(**kwargs)


class File(FSOutput):
    @classmethod
    def temp(cls, mode: str = "wb", **kwargs):
        tmp = NamedTemporaryFile(mode, **kwargs)
        ret = cls(tmp.name)
        ret.file = tmp.file
        ret._closer = tmp._closer
        return ret


class UntypedOuput(UntypedWarning): ...


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
        assert self.typ is not None
        if (have := self.is_dir) != (want := data.is_dir()):
            have = "directory" if have else "file"
            want = "directory" if want else "file"
            raise Incompatible(f"got {have} expected a {want}")
        if not isinstance(data, self.typ):
            warn(f"got a {type(data)}, expected a {self.typ}", Inconsistent)
            data = self.typ(data)
        return data


class Output[T](BaseOutput[T]):
    """
    Basic output implementation for local file storage.
        Uses pickle by default, but also supports
        JSON and direct File and Dir output.
    """

    base = "."

    @property
    def stem(self) -> str:
        return f"{self.act.name}_{hex_hash(self.act.instance).decode("ascii")}"

    @property
    def suffix(self) -> str:
        return self.coder.suffix

    @property
    def path(self) -> Path:
        return Path(self.base, self.stem).with_suffix(self.suffix)

    @property
    def coder(self) -> Coder[T]:
        typs = tuple(self._utypes)
        if not typs:
            UntypedOuput.warn(self.act.action, self.act.instance)
            return self._coders[-1]
        for enc in self._coders:
            for typ in typs:
                if enc.match(typ):
                    return enc
        raise NotSupported(f"no coder for {typ}")

    _coders: list[Coder] = [
        Coder(".dir", typ=Dir),
        Coder(".file", typ=File),
        Coder(".json", JSON, json.load, json.dump),
        Coder(".pkl", Pickle, pickle.load, pickle.dump, typ=object),
    ]

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
            dir=path.parent,
            prefix=path.with_suffix(".tmp").name,
            suffix=path.suffix,
        )

        if (is_dir := self.coder.is_dir) or mode == "":
            if mode or kwargs:
                raise ValueError("superfluous arguments!")
            ret = self._ftype.tempDir(**dps)
            return ret if is_dir else ret.joinpath(path.with_stem("temp").name)

        if mode is None:
            mode = "wb"

        return self._ftype.temp(mode, **dps, **kwargs)

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

    def load(self) -> T:
        if callable(load := self.coder.load):
            with self.path.open("rb") as f:
                return load(f)
        else:
            return self.type(self.path)

    def dump(self, data: T) -> T:
        if isinstance(data, FSOutput):
            if not data.exists():
                raise MissingOuput(str(data))
            data = self.coder.conform(data)
            if not data.temp or not data.is_relative_to(self.path.parent):
                if self.coder.is_dir:
                    data = copytree(data, self.temp(), dirs_exist_ok=True)
                else:
                    data = copy2(data, self.temp(""))
            assert data.temp
            if callable(close := getattr(data.file, "close", None)):
                close()
            data = data.replace(self.path)
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
            tmp = self.temp("wb")
            dump(data, tmp.file)
            self.dump(tmp)
            return data
        else:
            raise NotSupported(f"got {type(data)}, expected {self.type}")

    # type stuff
    @property
    def _ftype(self) -> Type[FSOutput]:
        for typ in self._utypes:
            if isinstance(typ, type) and issubclass(typ, FSOutput):
                return typ
        return Dir if self.coder.is_dir else File

    @property
    def _utypes(self) -> Iterable[type]:
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
    def type(self) -> type | Unset:  # this brick type in here
        action = self.act.action
        return getattr(action.func, "__annotations__", {}).get(
            "return", getattr(action, "return_type", UNSET)
        )
