import sys
from json import JSONDecodeError
from unittest.mock import patch

import pytest

from parao.cast import CastError
from parao.cli import (
    CLI,
    AmbiguousCandidate,
    CLIParser,
    MalformedCommandline,
    MultipleCandidates,
    NotAParaO,
    ParaONotFound,
    Sep,
    UnmatchedArguments,
    UnsupportedKeyType,
    UnusedArguments,
    UnusedOptions,
    ValueMissing,
    ValueUnexpected,
)
from parao.core import MissingParameterValue, Param, ParaO


class Outer1(ParaO):
    class Inner(ParaO):
        foo = Param[int](2)
        boo = Param[str]("inner")

    foo = Param[int](1)
    bar = Param[str]("outer")
    boo = Param[bool](None)


class Outer2(ParaO):
    class Inner(ParaO):
        pass

    class Inner2(ParaO):
        pass

    foo = Param[int]()
    bar = Param[str]()


class Outer3(ParaO):
    class Inner2(ParaO):
        pass


Outer3.__module__ += ".sub"
Outer3.Inner2.__module__ += ".sub"


class MultiWrap(ParaO):
    inner1 = Param[ParaO]()
    inner2 = Param[ParaO]()


plain_object = object()


def test_argv():
    argv = ["<script>", "Outer1"]
    with patch("sys.argv", argv):
        assert sys.argv == argv
        assert isinstance(CLI().run()[0], Outer1)


def test_plain():
    cli = CLI()

    a, b = cli.run(["Outer1", "Outer1.Inner"])
    assert isinstance(a, Outer1)
    assert isinstance(b, Outer1.Inner)
    with pytest.raises(ParaONotFound, match="DoesNotExist"):
        cli.run(["DoesNotExist"])
    with pytest.warns(AmbiguousCandidate):
        pytest.raises(ParaONotFound, lambda: cli.run(["Inner"]))
    with pytest.warns(MultipleCandidates):
        pytest.raises(ParaONotFound, lambda: cli.run(["Inner2"]))
    pytest.raises(ModuleNotFoundError, lambda: cli.run(["does_not_exist.Inner2"]))
    assert isinstance(cli.run(["sub.Inner2"])[0], Outer3.Inner2)
    with pytest.raises(NotAParaO):
        cli.run(["tests.test_cli:plain_object"])


def test_params():
    cli = CLI()

    assert cli.run(["Outer1", "--foo", "123"])[0].foo == 123
    assert cli.run(["Outer1", "--foo=123"])[0].foo == 123
    assert cli.run(["Outer1", "--Outer1.foo=123"])[0].foo == 123
    # other integer literals
    assert cli.run(["Outer1", "--foo", "0x10"])[0].foo == 0x10
    assert cli.run(["Outer1", "--foo", "0o10"])[0].foo == 0o10
    assert cli.run(["Outer1", "--foo", "0b10"])[0].foo == 0b10
    # various empties
    assert cli.run(["Outer1", "--bar="])[0].bar == ""
    assert cli.run(["Outer1"])[0].boo is None
    assert cli.run(["Outer1", "--boo"])[0].boo is True
    assert cli.run(["Outer1", "--boo=n"])[0].boo is False
    with pytest.raises(CastError):
        assert cli.run(["Outer1", "--boo=what?"])[0].boo
    assert cli.run(["Outer1", "--boo", "--bar=b"])[0].boo is True
    # class
    with pytest.raises(ValueMissing):
        cli.run(["MultiWrap", "--inner1,__class__="])
    with pytest.raises(ParaONotFound):
        cli.run(["MultiWrap", "--inner1,__class__=ThisDoesNotExist"])
    (wrap,) = cli.run(["MultiWrap", "--inner1,__class__=Outer1"])
    assert isinstance(wrap.inner1, Outer1)
    assert isinstance(wrap.inner2, ParaO)
    # json
    assert len(cli.run(["Outer1", "--foo;json", "[1,2,3]"])) == 3
    with pytest.raises(ValueMissing):
        cli.run(["Outer1", "--foo;json="])
    with pytest.raises(JSONDecodeError):
        cli.run(["Outer1", "--foo;json=]"])
    # python literals
    assert cli.run(["Outer1", "--foo;python", "0o123"])[0].foo == 0o123
    # with module
    assert cli.run(["Outer1", "--test_cli.Outer1.foo=123"])[0].foo == 123
    assert cli.run(["Outer1", "--test_cli:Outer1.foo=123"])[0].foo == 123
    with pytest.raises(ModuleNotFoundError):
        cli.run(["Outer1", "--test_cli:bad.Outer1.foo=123"])
    with pytest.warns(UnmatchedArguments):
        assert cli.run(["Outer1", "--not_found.foo=123"])[0].foo == 1
    assert cli.run(["Outer1", "--tests.test_cli:Outer1.foo=123"])[0].foo == 123
    with pytest.raises(MalformedCommandline):
        cli.run(["Outer1", "--tests.test_cli:=123"])
    with pytest.warns(UnsupportedKeyType), pytest.raises(TypeError):
        cli.run(["Outer1", "--tests.test_cli:plain_object=123"])
    # expansion
    assert len(cli.run(["Outer1", "--foo=1,2"])) == 2
    assert cli.run(["Outer1", "--bar=a,b"])[0].bar == "a,b"
    with pytest.raises(ValueError):
        cli.run(["Outer1", "--boo=0,1", "--foo=1,x"])


def test_global():
    # global agruments
    assert CLI(["--foo", "123"]).run(["Outer1"])[0].foo == 123
    with pytest.raises(TypeError):
        CLI([1])
    with pytest.raises(ValueUnexpected, match="foo"):
        CLI(["foo"])
    with pytest.raises(ValueUnexpected, match="--"):
        CLI(["--"])


def test_prio():
    cli = CLI()
    with pytest.warns(UnusedArguments):
        assert cli.run(["Outer1", "-foo=9", "-foo=1"])[0].foo == 1
        assert cli.run(["Outer1", "+foo=9", "-foo=1"])[0].foo == 9
        assert cli.run(["Outer1", "-+foo=9", "-foo=1"])[0].foo == 9
        assert cli.run(["Outer1", "-foo;prio:=9", "-foo=1"])[0].foo == 1
        assert cli.run(["Outer1", "-foo;prio:1=9", "-foo=1"])[0].foo == 9
        assert cli.run(["Outer1", "-foo;prio:1.1=9", "-foo=1"])[0].foo == 9
    with patch.object(Outer1.foo, "min_prio", 2):
        assert cli.run(["Outer1", "-foo=3"])[0].foo == 1
    with pytest.raises(ValueError):
        cli.run(["Outer1", "-foo;prio:x=9"])


def test_unused_arguments():
    cli = CLI()
    assert cli.run([]) == []
    cli.run(["", "Outer1", ""])
    with pytest.warns(UnusedOptions):
        cli.run(["--foo", "Outer1"])
    with pytest.warns(UnusedOptions):
        cli.run(["Outer1", "--", "--foo"])


def test_unused_parameters():
    cli = CLI()

    args_not_used = ["--not-used1", "--not-used2"]
    with pytest.warns(UnmatchedArguments, match=" ".join(args_not_used)):
        cli.run(["Outer1"] + args_not_used)

    arg_shadowed = "--foo=NOT@ACTIVE"
    with pytest.warns(UnusedArguments, match=arg_shadowed):
        cli.run(["Outer1", arg_shadowed, "--foo=3"])

    with (
        pytest.warns(UnmatchedArguments, match=" ".join(args_not_used)),
        pytest.warns(UnusedArguments, match=arg_shadowed),
    ):
        cli.run(["Outer1", arg_shadowed, *args_not_used, "--foo=3"])


def test_errors():
    with pytest.raises(MissingParameterValue):
        CLI().run(["Outer2"])
    with pytest.raises(MissingParameterValue), pytest.warns(UnusedArguments):
        CLI().run(["Outer2", "-foo", "1"])
    with pytest.raises(MissingParameterValue), pytest.warns(UnusedArguments):
        CLI().run(["Outer2", "-foo", "1,2"])


def test_sep():
    pytest.raises(Sep.NeedValues, lambda: Sep(()))
    pytest.raises(Sep.Overlap, lambda: Sep("x") << Sep("xy"))
    assert Sep(("foo", "bar")).regex.pattern == "(?:foo|bar)"
    assert Sep((*"foo", "bar")).regex.pattern == "(?:bar|[foo])"


def test_parser():
    p = CLIParser(flag="=")
    assert p._flag_value_disjoint is False
    assert p.argument("foo=json=val") == ([("foo", None)], {"json": None}, "val")
    assert p.argument("foo=") == ([("foo", None)], {"": None}, None)
    assert p.argument("foo") == ([("foo", None)], None, None)

    class Sub(CLIParser):
        extra: int = 0

    Sub()
