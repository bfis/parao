from json import JSONDecodeError
import sys
from unittest import TestCase
from unittest.mock import patch
from parao.cli import (
    CLI,
    AmbigouusCandidate,
    CLIParser,
    MalformedCommandline,
    MultipleCandidates,
    NotAParaO,
    ParaONotFound,
    Sep,
    UnsupportedKeyType,
    UnusedCLIArguments,
    ValueMissing,
)
from parao.core import MissingParameterValue, ParaO, Param


class Outer1(ParaO):
    class Inner(ParaO):
        foo = Param[int](2)
        boo = Param[str]("inner")

    foo = Param[int](1)
    bar = Param[str]("outer")
    boo = Param[bool](False)


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


class TestCLI(TestCase):
    def test_argv(self):
        argv = ["<script>", "Outer1"]
        with patch("sys.argv", argv):
            self.assertEqual(sys.argv, argv)
            self.assertIsInstance(CLI().run()[0], Outer1)

    def test_plain(self):
        cli = CLI()

        a, b = cli.run(["Outer1", "Outer1.Inner"])
        self.assertIsInstance(a, Outer1)
        self.assertIsInstance(b, Outer1.Inner)
        with self.assertRaises(ParaONotFound, msg="DoesNotExist"):
            cli.run(["DoesNotExist"])
        with self.assertWarns(AmbigouusCandidate):
            self.assertRaises(ParaONotFound, lambda: cli.run(["Inner"]))
        with self.assertWarns(MultipleCandidates):
            self.assertRaises(ParaONotFound, lambda: cli.run(["Inner2"]))
        self.assertRaises(
            ModuleNotFoundError, lambda: cli.run(["does_not_exist.Inner2"])
        )
        self.assertIsInstance(cli.run(["sub.Inner2"])[0], Outer3.Inner2)
        with self.assertRaises(NotAParaO):
            cli.run(["tests.test_cli:plain_object"])

    def test_params(self):
        cli = CLI()

        self.assertEqual(cli.run(["Outer1", "--foo", "123"])[0].foo, 123)
        self.assertEqual(cli.run(["Outer1", "--foo=123"])[0].foo, 123)
        self.assertEqual(cli.run(["Outer1", "--Outer1.foo=123"])[0].foo, 123)
        # various empties
        self.assertEqual(cli.run(["Outer1", "--bar="])[0].bar, "")
        self.assertEqual(cli.run(["Outer1", "--boo"])[0].boo, True)
        self.assertEqual(cli.run(["Outer1", "--boo", "--bar=b"])[0].boo, True)
        # class
        with self.assertRaises(ValueMissing):
            cli.run(["MultiWrap", "--inner1,__class__="])
        with self.assertRaises(ParaONotFound):
            cli.run(["MultiWrap", "--inner1,__class__=ThisDoesNotExist"])
        (wrap,) = cli.run(["MultiWrap", "--inner1,__class__=Outer1"])
        self.assertIsInstance(wrap.inner1, Outer1)
        self.assertIsInstance(wrap.inner2, ParaO)
        # json
        self.assertEqual(len(cli.run(["Outer1", "--foo;json", "[1,2,3]"])), 3)
        with self.assertRaises(ValueMissing):
            cli.run(["Outer1", "--foo;json="])
        with self.assertRaises(JSONDecodeError):
            cli.run(["Outer1", "--foo;json=]"])
        # with module
        self.assertEqual(cli.run(["Outer1", "--test_cli.Outer1.foo=123"])[0].foo, 123)
        self.assertEqual(cli.run(["Outer1", "--test_cli:Outer1.foo=123"])[0].foo, 123)
        with self.assertRaises(ModuleNotFoundError):
            cli.run(["Outer1", "--test_cli:bad.Outer1.foo=123"])
        self.assertEqual(cli.run(["Outer1", "--not_found.foo=123"])[0].foo, 1)
        self.assertEqual(
            cli.run(["Outer1", "--tests.test_cli:Outer1.foo=123"])[0].foo, 123
        )
        with self.assertRaises(MalformedCommandline):
            cli.run(["Outer1", "--tests.test_cli:=123"])
        with self.assertWarns(UnsupportedKeyType), self.assertRaises(TypeError):
            cli.run(["Outer1", "--tests.test_cli:plain_object=123"])
        # expansion
        self.assertEqual(len(cli.run(["Outer1", "--foo=1,2"])), 2)
        with self.assertRaises(ValueError):
            cli.run(["Outer1", "--bar=A,B", "--foo=1,x"])

    def test_prio(self):
        cli = CLI()
        self.assertEqual(cli.run(["Outer1", "-foo=9", "-foo=1"])[0].foo, 1)
        self.assertEqual(cli.run(["Outer1", "+foo=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Outer1", "-+foo=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Outer1", "-foo;prio:=9", "-foo=1"])[0].foo, 1)
        self.assertEqual(cli.run(["Outer1", "-foo;prio:1=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Outer1", "-foo;prio:1.1=9", "-foo=1"])[0].foo, 9)
        with patch.object(Outer1.foo, "min_prio", 2):
            self.assertEqual(cli.run(["Outer1", "-foo=3"])[0].foo, 1)
        with self.assertRaises(ValueError):
            cli.run(["Outer1", "-foo;prio:x=9"])

    def test_unused_arguments(self):
        cli = CLI()
        self.assertEqual(cli.run([]), [])
        cli.run(["", "Outer1", ""])
        with self.assertWarns(UnusedCLIArguments):
            cli.run(["--foo", "Outer1"])
        with self.assertWarns(UnusedCLIArguments):
            cli.run(["Outer1", "--", "--foo"])

    def test_errors(self):
        self.assertRaises(MissingParameterValue, lambda: CLI().run(["Outer2"]))
        self.assertRaises(
            MissingParameterValue, lambda: CLI().run(["Outer2", "-foo", "1"])
        )
        self.assertRaises(
            MissingParameterValue, lambda: CLI().run(["Outer2", "-foo", "1,2"])
        )


class TestSep(TestCase):
    def test(self):
        self.assertRaises(Sep.NeedValues, lambda: Sep(()))
        self.assertRaises(Sep.Overlap, lambda: Sep("x") << Sep("xy"))
        self.assertEqual(Sep(("foo", "bar")).regex.pattern, "(?:foo|bar)")
        self.assertEqual(Sep((*"foo", "bar")).regex.pattern, "(?:bar|[foo])")


class TestCLIParser(TestCase):
    def test(self):
        p = CLIParser(flag="=")
        self.assertFalse(p._flag_value_disjoint)
        self.assertEqual(
            p.argument("foo=json=val"),
            ([("foo", None)], {"json": None}, "val"),
        )
        self.assertEqual(
            p.argument("foo="),
            ([("foo", None)], {"": None}, None),
        )
        self.assertEqual(
            p.argument("foo"),
            ([("foo", None)], None, None),
        )

        class Sub(CLIParser):
            extra: int = 0

        Sub()
