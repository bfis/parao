import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from io import IOBase
from pathlib import Path
from stat import S_IMODE, S_IWGRP, S_IWOTH, S_IWUSR
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, call, patch
from warnings import catch_warnings

import pytest

from parao.action import ValueAction
from parao.cli import CLI
from parao.core import Arguments, Const, OwnParameters, Param, ParaO
from parao.output import (
    JSON,
    Coder,
    Dir,
    FancyTemplate,
    File,
    FSOutput,
    Inconsistent,
    MissingOuput,
    MoveAcrossFilesystem,
    NotSupported,
    Output,
    Pickle,
    RunAct,
    UntypedOuput,
)
from parao.run import ConcurrentRunner, PseudoOutput
from parao.task import RunAction, Task, pprint

sentinel = object()


class BaseTask[T](Task[T]):
    space_waster = Param[str]("")


class Task1(BaseTask):
    code_version = 1

    @RunAction[tuple[int, str]]
    def run(self): ...

    run.aux = sentinel


class Task2(BaseTask):
    dep = Param[Task1]()

    def run(self) -> tuple:
        return 321, self.dep.run.output.load()


@pytest.fixture
def tmpdir4BaseTask(tmpdir):
    with Arguments.context(Arguments.make({(FancyTemplate, "dir_base"): tmpdir})):
        yield tmpdir


def test(tmpdir4BaseTask):
    return_sentinel = 123, "foo"
    mock = Mock(return_value=return_sentinel)

    with patch.object(Task1.run, "func", mock):
        assert Task1.run.type is RunAction.type

        assert isinstance(Task1.code_version, Const)
        assert Task1.code_version.value == 1
        assert isinstance(Task1.__own_parameters__["code_version"], Const)
        assert isinstance(Task1.run, RunAction)
        assert Task1.run.aux is sentinel

        t1 = Task1()

        assert isinstance(t1.run, RunAct)
        assert t1.run.action.return_type == tuple[int, str]
        assert t1.run._key == (t1, Task1.run)

        assert t1.run.output is t1.run.output
        assert not t1.run.output.path.exists()
        assert not t1.run.output.exists
        assert not t1.run.done
        with pytest.raises(AttributeError):
            t1.run.does_not_exist

        assert t1.run() is return_sentinel
        mock.assert_called_once_with(t1)
        assert t1.run() == return_sentinel
        assert t1.run() is not return_sentinel  # not cached (yet!?)
        mock.assert_called_once_with(t1)  # still just once
        assert t1.run.done
        assert t1.run.output.load() == (123, "foo")
        assert t1.run.output.exists
        assert t1.run.output.path.exists()

        t2 = Task2()
        assert t2.run() == (321, (123, "foo"))
        assert t2.run.output.type is tuple

        t1.run.output.remove()
        assert not t1.run.done
        t2.run.output.remove()
        assert not t2.run.done

        assert t2.run() == (321, (123, "foo"))


# test outpus stuff
class TaskX(BaseTask):
    @RunAction
    def run(self): ...


def test_output_untyped(tmpdir4BaseTask):
    t = TaskX()
    with pytest.warns(UntypedOuput):
        t.run.output.coder


@pytest.fixture
def typedTaskX(tmpdir4BaseTask, request):
    with patch.object(TaskX.run, "return_type", request.param, create=True):
        yield TaskX()


class UnsupportedPseudoOutput(PseudoOutput): ...


@pytest.mark.parametrize(
    "typedTaskX",
    [None, str | int, JSON, JSON[list], Pickle, Pickle[list]],
    indirect=True,
)
def test_output_transparent(typedTaskX):
    tmp = typedTaskX.run.output.temp()
    assert isinstance(tmp, File)
    assert isinstance(tmp.tmpio, IOBase)
    assert tmp._temp is None


def test_output_unknown(tmpdir4BaseTask):
    with (
        patch.object(TaskX.output.type._coders[-1], "typ", None),
        patch.object(TaskX.run.func, "__annotations__", {"return": None}),
        pytest.raises(NotSupported),
    ):
        TaskX().run.output.coder

    with patch.object(TaskX.run, "func") as mock:
        TaskX.run.func.__annotations__ = {"return": UnsupportedPseudoOutput}
        mock.return_value = UnsupportedPseudoOutput()
        with pytest.raises(NotSupported):
            TaskX().run()


def test_output_json(tmpdir4BaseTask):
    with (
        patch.object(TaskX.run, "return_type", JSON, create=True),
        patch.object(TaskX.run, "func") as mock,
    ):
        mock.return_value = [1, 2, 3]
        t = TaskX()
        assert t.run() is mock.return_value
        assert t.run.output.load() == mock.return_value


def test_output_extraCoder(tmpdir4BaseTask):
    type JSON2[T] = T

    coders_extra = (
        Coder(".json", JSON2, json.load, partial(json.dump, indent=2), text=True),
    )

    with (
        patch.object(TaskX.run, "return_type", JSON2, create=True),
        patch.object(TaskX.run, "func") as mock,
    ):
        mock.return_value = [1, 2, 3]
        t = TaskX({(FancyTemplate, "coders_extra"): coders_extra})
        assert t.run() is mock.return_value
        assert t.run.output.path.read_text() == "[\n  1,\n  2,\n  3\n]"
        assert t.run.output.load() == mock.return_value


@contextmanager
def make_readonly(target):
    target = Path(target)
    stat = S_IMODE(target.stat().st_mode)
    target.chmod(stat & ~(S_IWUSR | S_IWGRP | S_IWOTH))
    try:
        yield
    finally:
        target.chmod(stat)


def test_output_other_temp(tmpdir4BaseTask):
    with patch.object(TaskX.run, "func") as mock:
        TaskX.run.func.__annotations__ = {"return": Dir}

        other_fs = os.environ.get(
            "TEMP2_ON_DIFFERENT_FS", os.environ.get("XDG_RUNTIME_DIR", None)
        )
        if (
            other_fs and os.stat(tmpdir4BaseTask).st_dev != os.stat(other_fs).st_dev
        ):  # pragma: no branch
            mock.return_value = Dir.temp(dir=other_fs)
            with (
                pytest.warns(MoveAcrossFilesystem, match="slow"),
                pytest.warns(MoveAcrossFilesystem, match="failed"),
            ):
                TaskX().run()
            TaskX().run.output.remove()
            mock.assert_called_once()
            mock.reset_mock()

        tmp = Dir.temp()
        mock.return_value = tmp.parent / tmp.name
        TaskX().run()
        TaskX().run.output.remove()
        mock.assert_called_once()
        mock.reset_mock()

        # now files
        TaskX.run.func.__annotations__ = {"return": File}

        run = TaskX().run

        tmp = run.output.temp()
        mock.return_value = tmp.parent / tmp.name
        run()
        run.output.remove()
        mock.assert_called_once()
        mock.reset_mock()

        mock.return_value = tmp = run.output.temp()
        with (
            make_readonly(tmp.parent),
            pytest.raises(OSError),
        ):  # trigger failure in rename
            run()
        mock.assert_called_once()
        mock.reset_mock()


class TaskDir(BaseTask):
    def run(self) -> Dir:
        ret: Dir = self.run.output.temp()
        ret.joinpath("foo.bar").touch()
        return ret


def test_output_Dir_alt(tmpdir4BaseTask):
    class DirAlt(Dir): ...

    with patch.object(TaskDir.run, "func") as mock:
        TaskDir.run.func.__annotations__ = {"return": DirAlt}
        mock.return_value = Dir.temp()
        TaskDir().run()
        mock.assert_called_once()


def test_output_Dir_bad(tmpdir4BaseTask):
    with patch.object(TaskDir.run, "func") as mock:
        TaskDir.run.func.__annotations__ = {"return": Dir}

        mock.return_value = None
        with pytest.raises(NotSupported):
            TaskDir().run()
        mock.assert_called_once()
        mock.reset_mock()

        o = TaskDir().run.output.temp()
        assert isinstance(o, Dir)
        o = FSOutput(o)
        assert isinstance(o, FSOutput)
        assert not isinstance(o, Dir)
        assert o._temp
        mock.return_value = o
        with pytest.warns(Inconsistent):
            TaskDir().run()
        TaskDir().remove()
        mock.assert_called_once()
        mock.reset_mock()

        o = TaskDir().run.output.temp() / "not_existing"
        assert isinstance(o, Dir)
        mock.return_value = o
        with pytest.raises(MissingOuput):
            TaskDir().run()
        mock.assert_called_once()
        mock.reset_mock()

        f: File = TaskDir().run.output.temp().joinpath("some.file")
        f.touch()
        assert f.is_file()
        mock.return_value = f
        with pytest.raises(IsADirectoryError):
            TaskDir().run()
        mock.assert_called_once()
        mock.reset_mock()


def test_output_Dir_temp(tmpdir4BaseTask):
    out: Output[Dir] = TaskDir().run.output
    with pytest.raises(ValueError):
        out.temp("foo")
    with pytest.raises(ValueError):
        out.temp(anything=True)

    tmp = out.temp()
    assert isinstance(tmp, Dir)
    assert isinstance(tmp._temp, TemporaryDirectory)

    sub = tmp / "inner.file"
    assert isinstance(sub, Dir)
    assert isinstance(sub._temp, TemporaryDirectory)
    assert tmp._temp is sub._temp

    parent = tmp.parent
    assert isinstance(parent, Dir)
    assert parent._temp is None


def test_output_Dir_remove(tmpdir4BaseTask):
    td = TaskDir()
    assert isinstance(td.run(), Dir)
    assert isinstance(td.run(), Dir)  # yes, again!
    assert isinstance(td.run.output.load(), Dir)

    d = td.run()
    assert d.is_dir()
    assert d.joinpath("foo.bar").is_file()
    td.run.output.remove()
    assert not d.exists()

    td.run.output.remove(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        td.run.output.remove(missing_ok=False)
    with pytest.raises(FileNotFoundError):
        td.run.output.remove()


def test_bad_output():
    with catch_warnings(action="ignore", category=OwnParameters.CacheReset):

        class Foo(Task):
            def run(): ...

            output = None

        with pytest.raises(TypeError):
            Foo().run.output


def test_print(capsys):
    with patch.object(pprint, "_stream", sys.stdout):
        Task2().print()

        cap = capsys.readouterr()
        assert cap.err == ""
        assert cap.out.split("\n") == [
            "tests.test_task:Task2(dep=tests.test_task:Task1())",
            "  tests.test_task:Task1()",
            "",
        ]


def test_status(capsys, tmpdir4BaseTask):
    with patch.object(pprint, "_stream", sys.stdout):
        Task2().status()

        cap = capsys.readouterr()
        assert cap.err == ""
        assert cap.out.split("\n") == [
            "tests.test_task:Task2(dep=tests.test_task:Task1()): missing",
            "  tests.test_task:Task1(): missing",
            "",
        ]

        filler = "~".join(map(str, range(50)))

        Task2({(Task, "space_waster"): filler}).status()

        cap = capsys.readouterr()
        assert cap.err == ""
        assert cap.out.split("\n") == [
            "tests.test_task:Task2(",
            " dep=tests.test_task:Task1(...),",
            f" space_waster='{filler}'",
            "): missing",
            "  tests.test_task:Task1(",
            f"   space_waster='{filler}'",
            "  ): missing",
            "",
        ]


def test_remove(capsys, tmpdir4BaseTask):
    with patch.object(pprint, "_stream", sys.stdout):
        t2 = Task2()
        t2.run()

        t2.remove(0)
        cap = capsys.readouterr()
        assert cap.err == ""
        assert cap.out.split("\n") == [
            "tests.test_task:Task2(dep=tests.test_task:Task1()): removed",
            "",
        ]

        t2.remove()
        cap = capsys.readouterr()
        assert cap.err == ""
        assert cap.out.split("\n") == [
            "tests.test_task:Task2(dep=tests.test_task:Task1()): missing",
            "  tests.test_task:Task1(): removed",
            "",
        ]


def test_action_ordering():
    func = Mock()

    class Foo(ParaO):
        act1 = ValueAction[int, None](func)
        act2 = ValueAction[int, None](func)

    cli = CLI(entry_points=[Foo])

    [foo] = cli.run(["Foo", "--act1=1", "--act2=2"])
    func.assert_has_calls([call(foo, 1), call(foo, 2)])

    func.reset_mock()

    [foo] = cli.run(["Foo", "--act2=2", "--act1=1"])
    func.assert_has_calls([call(foo, 2), call(foo, 1)])


def test_ConcurrentRunner(tmpdir4BaseTask):
    runner = ConcurrentRunner(ThreadPoolExecutor(1))

    task = Task2()
    res = task.run()

    task.remove()
    assert task.run(runner=runner) == res
    assert task.run.done

    task.remove()
    with ConcurrentRunner.current(runner):
        assert task.run() == res
        assert task.run() == res  # yes twice
        assert task.run.done


class TaskC(Task):
    in1 = Param[Any](None, neutral=None, pos=1)
    in2 = Param[Any](None, neutral=None, pos=2)
    in3 = Param[Any](None, neutral=None, pos=3)
    in4 = Param[Any](None, neutral=None, pos=4)

    def run(self) -> None: ...


class TaskL(Task):
    common = Param[TaskC]()
    different = Param[str]("left")


class TaskR(Task):
    common = Param[TaskC]()
    different = Param[str]("right")


class TaskT(Task):
    def run(self) -> None: ...

    labeled = Param[int](0, pos=0)

    small11 = Param[int](11, pos=1.1)
    small12 = Param[int](12, pos=1.2)
    small13 = Param[int](13, pos=1.3)

    normal = Param[int](2, pos=2)

    small21 = Param[int](21, pos=2.1)
    small22 = Param[int](22, pos=2.2)
    small23 = Param[int](23, pos=2.3)

    left = Param[TaskL](pos=5)
    right = Param[TaskR](pos=5)
    left2 = Param[TaskL]()


def test_templating(tmpdir):
    altdir = tmpdir / "_alt"
    r = Task1(
        {
            (FancyTemplate, "dir_base"): tmpdir,
            (FancyTemplate, "dir_temp"): altdir,
        }
    ).run
    assert r.output.path.is_relative_to(tmpdir)
    assert r.output.temp().is_relative_to(altdir)

    def probe(task: type[Task], extra: dict = {}, /, **kwargs) -> tuple[str]:
        kwargs["dir_base"] = tmpdir
        t = task(extra, {(FancyTemplate, k): v for k, v in kwargs.items()})
        return t.run.output.path.relative_to(tmpdir).parts[:-1]

    assert probe(TaskT) == (
        "tests.test_task:TaskT",
        "labeled=0",
        "11_12_13",
        "2",
        "21_22_23",
    )
    assert probe(TaskT, tot_limit=0) == probe(TaskT)
    assert probe(TaskT, tot_limit=40) == (
        "tests.test_task:TaskT",
        "labeled=0",
    )
    assert probe(TaskT, class_name=False, module_name=False) == (
        "labeled=0",
        "11_12_13",
        "2",
        "21_22_23",
    )
    assert probe(TaskT, dir_limit=1) == ("tests.test_task:TaskT",)
    assert probe(TaskT, dir_limit=-1) == ()
    assert probe(TaskT, name_limit=3, name_ellipsis="~") == (
        "te~",
        "la~",
        "11~",
        "2",
        "21~",
    )

    assert probe(TaskT, {"in1": "foo"}, default_pos=1e3 + 0.1) == (
        "tests.test_task:TaskT",
        "labeled=0",
        "11_12_13",
        "2",
        "21_22_23",
        "foo",
        "left",
        "right",
    )

    with patch.object(os, "pathconf", lambda p, n: os.doesnotexist):
        assert probe(
            TaskC,
            name_limit=-(4 + len(os.fsencode(FancyTemplate.name_ellipsis.default))),
        ) == ("test…",)

    assert probe(TaskC, {"in1": "………"}, name_limit=5) == ("te…", "…")
    assert probe(
        TaskC,
        {
            "in1": {
                0: None,
                1: True,
                2: 0.2,
                3: 3j,
                4: "four",
                5: b"bytes",
                6: ...,
            },
            "in2": [1, "2", 3.0],
            "in3": set("cab"),
        },
    ) == (
        "tests.test_task:TaskC",
        "0:None,1:True,2:0.2,3:3j,4:four,5:6279746573,6:Ellipsis",
        "1,2,3.0",
        "a,b,c",
    )
