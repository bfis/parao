from unittest import TestCase
from parao.core import Arg, Arguments, ParaO, Param, MissingParameterValue, srepr


uniq_object = object()


class TestArg(TestCase):

    def test_create(self):
        key = "foo", "bar", "boo"
        a = Arg(key, uniq_object)
        self.assertEqual(a.key, key)
        self.assertEqual(a.val, uniq_object)
        self.assertEqual(a.prio, 0)
        self.assertEqual(a.offset, 0)
        # prio argument
        self.assertEqual(Arg(key, uniq_object, 123).prio, 123)
        # offset argument
        self.assertEqual(Arg(key, uniq_object, 0, 123).offset, 123)

    def test_repr(self):

        self.assertEqual(
            repr(Arg(("foo", "bar"), val=123, prio=321)),
            "Arg('foo', 'bar', val=123, prio=321)",
        )
        self.assertEqual(
            repr(Arg(("foo", "bar"), val=123, offset=1)),
            "Arg('bar', val=123)",
        )


class TestArguments(TestCase):

    def test_create(self):
        tpl = (1, "foo", uniq_object)  # the are actually bad types ...
        self.assertTupleEqual(Arguments(tpl), tpl)

        self.assertEqual(
            Arguments(
                [
                    Arg(("foo",), uniq_object, prio=123),
                    Arg(("foo", "bar"), uniq_object, prio=123),
                ]
            ),
            Arguments.from_dict(
                {"foo": uniq_object, ("foo", "bar"): uniq_object},
                prio=123,
            ),
        )
        self.assertEqual(
            Arguments(
                [
                    Arg(("foo",), uniq_object, prio=123),
                    Arg(("foo", "bar"), uniq_object, prio=123),
                ]
            ),
            Arguments.from_dict(
                [
                    ("foo", uniq_object),
                    (("foo", "bar"), uniq_object),
                ],
                prio=123,
            ),
        )

        self.assertIs(Arguments.from_list([]), Arguments.EMPTY)
        self.assertIs(Arguments.from_list([a := Arguments()]), a)
        self.assertEqual(
            Arguments.from_list([Arg(("foo",), uniq_object)]),
            Arguments([Arg(("foo",), uniq_object)]),
        )

        self.assertRaises(TypeError, lambda: ParaO(123))

    def test_repr(self):

        self.assertEqual(repr(Arguments.make()), "Arguments()")
        self.assertEqual(
            repr(Arguments.make(key=123)),
            "Arguments(Arg('key', val=123),)",
        )
        self.assertEqual(
            repr(Arguments.make(foo=123, bar=456)),
            "Arguments(Arg('foo', val=123), Arg('bar', val=456))",
        )


class TestParam(TestCase):
    def test_param(self):
        self.assertIs(Param(type=(o := object())).type, o)
        self.assertIs(Param[o := object()]().type, o)
        self.assertRaises(TypeError, lambda: Param[int, str])


class TestParaO(TestCase):
    def test_create(self):
        ParaO()

        class Sub(ParaO): ...

        self.assertIsInstance(Sub(), Sub)
        self.assertIsInstance(ParaO({ParaO: Sub}), Sub)
        self.assertIsInstance(ParaO({"__class__": Sub}), Sub)

        self.assertRaises(TypeError, lambda: ParaO({ParaO: 123}))

    def test_own_params(self):
        class Sub(ParaO):
            foo: int = Param()
            bar: str = Param()

        self.assertEqual(Sub.__own_parameters__, {"foo": Sub.foo, "bar": Sub.bar})

        Sub.boo = Param(type=float)

        self.assertEqual(Sub.__own_parameters__["boo"], Sub.boo)

        Sub.boo = Param(type=complex)

        self.assertEqual(Sub.__own_parameters__["boo"], Sub.boo)

        del Sub.foo, Sub.bar, Sub.boo

        self.assertEqual(Sub.__own_parameters__, {})

    def test_resolution_simple(self):

        class Sub(ParaO):
            foo: int = Param()
            bar = Param(None, type=str)
            boo = Param[bool]()

        self.assertEqual(Sub.boo.type, bool)

        with self.assertRaises(MissingParameterValue):
            Sub().foo
        self.assertEqual(Sub({"foo": 123}).foo, 123)
        self.assertEqual(Sub({Sub.foo: 123}).foo, 123)
        self.assertEqual(Sub({(Sub, "foo"): 123}).foo, 123)
        self.assertEqual(Sub({(Sub, Sub, "foo"): 123}).foo, 123)
        self.assertEqual(Sub({(Sub, Sub.foo): 123}).foo, 123)

        self.assertEqual(Sub().bar, None)
        self.assertEqual(Sub(bar=123).bar, "123")

    def test_resolution_complex(self):

        class Sub(ParaO):
            foo: int = Param()
            bar: str = Param(None)

        class Sub2(Sub):
            boo: bool = Param()

        self.assertEqual(Sub({Sub: Sub2, "boo": True}).boo, True)

        class Wrap(ParaO):
            one: Sub = Param()
            other: Sub2 = Param()

        class More(ParaO):
            inner: Wrap = Param()

        for addr in [("one", "bar"), (Wrap.one, Sub.bar), (Wrap.one, Sub, "bar")]:
            with self.subTest(addr=addr):
                self.assertEqual(Wrap({addr: 123}).one.bar, "123")
                self.assertEqual(Wrap({addr: 123}).other.bar, None)

        # providing a dict
        self.assertEqual(Wrap(one=dict(foo=123)).one.foo, 123)

        # unsing instance's args
        self.assertEqual(Wrap(one=Sub2(foo=123)).one.foo, 123)
        self.assertEqual(Wrap(one=Sub2(foo=123).__args__).one.foo, 123)

        # direct instance providing
        self.assertEqual(Wrap(one=Sub(foo=123)).one.foo, 123)
        self.assertIs(Wrap(one=(s := Sub())).one, s)

        # self.assertEqual(More().inner)

        # obj = Wrap({(Sub, "foo"): 123})
        # self.assertEqual(obj.one.foo, 123)
        # self.assertEqual(obj.other.foo, 123)


class TestMisc(TestCase):
    def test_srepr(self):

        class Foo:
            def __repr__(self):
                raise RuntimeError()

        self.assertRaises(RuntimeError, lambda: repr(Foo()))
        self.assertEqual(srepr(o := Foo()), object.__repr__(o))
