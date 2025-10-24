from io import IOBase
import sys
from tempfile import TemporaryDirectory
import pytest
from unittest.mock import Mock, patch

from parao.core import Const, Param
from parao.output import (
    JSON,
    Dir,
    FSOutput,
    File,
    Incompatible,
    Inconsistent,
    MissingOuput,
    NotSupported,
    Pickle,
    UntypedOuput,
)
from parao.run import PseudoOutput, RunAct
from parao.task import RunAction, Task, Output, pprint


sentinel = object()


class BaseTask[T](Task[T]):
    space_waster = Param[str]("")

    class output(Output):
        base = None


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
    with patch.object(BaseTask.output, "base", tmpdir):
        yield


def test(tmpdir4BaseTask):
    return_sentinel = 123, "foo"
    mock = Mock(return_value=return_sentinel)

    with patch.object(Task1.run, "func", mock):
        assert issubclass(Task1.output, Output)

        assert Task1.run.type is RunAction.type

        assert isinstance(Task1.code_version, Const)
        assert Task1.code_version.value == 1
        assert isinstance(Task1.__own_parameters__["code_version"], Const)
        assert isinstance(Task1.run, RunAction)
        assert Task1.run.aux is sentinel

        t1 = Task1()

        assert isinstance(t1.run, RunAct)
        assert t1.run.action.return_type == tuple[int, str]

        assert not t1.run.output.path.exists()
        assert not t1.run.output.exists
        assert not t1.run.done

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
    assert isinstance(tmp.file, IOBase)
    assert tmp._temp is None


def test_output_unknown(tmpdir4BaseTask):
    run = TaskX().run
    with (
        patch.object(run.output._coders[-1], "typ", None),
        patch.object(TaskX.run.func, "__annotations__", {"return": None}),
        pytest.raises(NotSupported),
    ):
        run.output.coder

    with patch.object(TaskX.run, "func") as mock:
        TaskX.run.func.__annotations__ = {"return": UnsupportedPseudoOutput}
        mock.return_value = UnsupportedPseudoOutput()
        with pytest.raises(NotSupported):
            TaskX().run()


def test_output_other_temp(tmpdir4BaseTask):
    with patch.object(TaskX.run, "func") as mock:
        TaskX.run.func.__annotations__ = {"return": Dir}
        mock.return_value = Dir.temp()
        TaskX().run()
        TaskX().run.output.remove()
        mock.assert_called_once()
        mock.reset_mock()

        TaskX.run.func.__annotations__ = {"return": File}
        mock.return_value = File.temp()
        TaskX().run()
        TaskX().run.output.remove()
        mock.assert_called_once()
        mock.reset_mock()


class TaskDir(BaseTask):
    def run(self) -> Dir:
        ret: Dir = self.run.output.temp()
        ret.joinpath("foo.bar").touch()
        return ret


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
        with pytest.raises(Incompatible):
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
