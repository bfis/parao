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
)
from parao.core import MissingParameterValue, ParaO, Param


class Test1(ParaO):
    class Inner(ParaO):
        foo = Param[int](2)
        boo = Param[str]("inner")

    foo = Param[int](1)
    bar = Param[str]("outer")
    boo = Param[bool](False)


class TestCLI1sub(Test1):
    extra = Param[complex](0)


class Test2(ParaO):
    class Inner(ParaO):
        pass

    class Inner2(ParaO):
        pass

    foo = Param[int]()
    bar = Param[str]()


class Test3(ParaO):
    class Inner2(ParaO):
        pass


Test3.__module__ += ".sub"
Test3.Inner2.__module__ += ".sub"

plain_object = object()


class TestCLI(TestCase):
    def test_argv(self):
        argv = ["<script>", "Test1"]
        with patch("sys.argv", argv):
            self.assertEqual(sys.argv, argv)
            self.assertIsInstance(CLI().run()[0], Test1)

    def test_plain(self):
        cli = CLI()

        a, b = cli.run(["Test1", "Test1.Inner"])
        self.assertIsInstance(a, Test1)
        self.assertIsInstance(b, Test1.Inner)
        with self.assertRaises(ParaONotFound, msg="DoesNotExist"):
            cli.run(["DoesNotExist"])
        with self.assertWarns(AmbigouusCandidate):
            self.assertRaises(ParaONotFound, lambda: cli.run(["Inner"]))
        with self.assertWarns(MultipleCandidates):
            self.assertRaises(ParaONotFound, lambda: cli.run(["Inner2"]))
        self.assertRaises(
            ModuleNotFoundError, lambda: cli.run(["does_not_exist.Inner2"])
        )
        self.assertIsInstance(cli.run(["sub.Inner2"])[0], Test3.Inner2)
        with self.assertRaises(NotAParaO):
            cli.run(["tests.test_cli:plain_object"])

    def test_params(self):
        cli = CLI()

        self.assertEqual(cli.run(["Test1", "--foo", "123"])[0].foo, 123)
        self.assertEqual(cli.run(["Test1", "--foo=123"])[0].foo, 123)
        self.assertEqual(cli.run(["Test1", "--Test1.foo=123"])[0].foo, 123)
        # various empties
        self.assertEqual(cli.run(["Test1", "--bar="])[0].bar, "")
        self.assertEqual(cli.run(["Test1", "--boo"])[0].boo, True)
        self.assertEqual(cli.run(["Test1", "--boo", "--bar=b"])[0].boo, True)
        # json
        self.assertEqual(len(cli.run(["Test1", "--foo;json", "[1,2,3]"])), 3)
        with self.assertRaises(JSONDecodeError):
            cli.run(["Test1", "--foo;json=]"])
        # with module
        self.assertEqual(cli.run(["Test1", "--test_cli.Test1.foo=123"])[0].foo, 123)
        self.assertEqual(cli.run(["Test1", "--test_cli:Test1.foo=123"])[0].foo, 123)
        with self.assertRaises(ModuleNotFoundError):
            cli.run(["Test1", "--test_cli:bad.Test1.foo=123"])
        self.assertEqual(cli.run(["Test1", "--not_found.foo=123"])[0].foo, 1)
        self.assertEqual(
            cli.run(["Test1", "--tests.test_cli:Test1.foo=123"])[0].foo, 123
        )
        with self.assertRaises(MalformedCommandline):
            cli.run(["Test1", "--tests.test_cli:=123"])
        with self.assertWarns(UnsupportedKeyType):
            cli.run(["Test1", "--tests.test_cli:plain_object=123"])

    def test_prio(self):
        cli = CLI()
        self.assertEqual(cli.run(["Test1", "-foo=9", "-foo=1"])[0].foo, 1)
        self.assertEqual(cli.run(["Test1", "+foo=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Test1", "-+foo=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Test1", "-foo;prio:=9", "-foo=1"])[0].foo, 1)
        self.assertEqual(cli.run(["Test1", "-foo;prio:1=9", "-foo=1"])[0].foo, 9)
        self.assertEqual(cli.run(["Test1", "-foo;prio:1.1=9", "-foo=1"])[0].foo, 9)
        with self.assertRaises(ValueError):
            cli.run(["Test1", "-foo;prio:x=9"])

    def test_unused_arguments(self):
        cli = CLI()
        self.assertEqual(cli.run([]), [])
        cli.run(["", "Test1", ""])
        with self.assertWarns(UnusedCLIArguments):
            cli.run(["--foo", "Test1"])
        with self.assertWarns(UnusedCLIArguments):
            cli.run(["Test1", "--", "--foo"])

    def test_errors(self):
        self.assertRaises(MissingParameterValue, lambda: CLI().run(["Test2"]))
        self.assertRaises(
            MissingParameterValue, lambda: CLI().run(["Test2", "-foo", "1"])
        )
        self.assertRaises(
            MissingParameterValue, lambda: CLI().run(["Test2", "-foo", "1,2"])
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
