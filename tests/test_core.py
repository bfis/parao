from operator import attrgetter
import pickle
from unittest import TestCase
from unittest.mock import Mock
from warnings import catch_warnings
from parao.core import (
    UNSET,
    OwnParameters,
    AbstractParam,
    Arguments,
    Const,
    Expansion,
    Fragment,
    ParaO,
    Param,
    MissingParameterValue,
    Prop,
    TypedAlias,
    Unset,
    UntypedParameter,
    Value,
    eager,
    DuplicateParameter,
    ExpansionGeneratedKeyMissingParameter,
)
import pytest

uniq_object = object()


def test_unset():
    assert isinstance(UNSET, Unset)
    assert UNSET is Unset()
    assert Unset() is Unset()
    assert repr(UNSET) == "<UNSET>"


def test_Value():
    val = Value(uniq_object)
    assert val.val is uniq_object
    assert val.prio == 0
    assert val.position == 0

    assert repr(Value(None, prio=1)) == "Value(None, 1)"
    assert repr(Value(None, position=1)) == "Value(None, 0, 1)"
    assert repr(Value(None, prio=1, position=1)) == "Value(None, 1, 1)"


def test_Fragment():
    key = ("foo", "bar")
    f = Fragment.make(key, Value(uniq_object))
    assert f.param == key[0]
    assert f.types is None
    assert isinstance(f.inner, Fragment)
    i = f.inner
    assert i.param == key[1]
    assert i.types is None
    assert isinstance(i.inner, Value)

    assert (
        repr(Fragment.make(key, Value(None)))
        == "Fragment('foo', None, Fragment('bar', None, Value(None)))"
    )


def test_Arguments():
    tpl = (1, "foo", uniq_object)  # the are actually bad types ...
    assert Arguments(tpl) == tpl

    assert (
        Arguments(
            [
                Fragment.make(("foo",), Value(uniq_object, 123)),
                Fragment.make(("foo", "bar"), Value(uniq_object, 123)),
            ]
        )
        == Arguments.from_dict(
            {"foo": uniq_object, ("foo", "bar"): uniq_object},
            prio=123,
        )
        == Arguments.from_dict(
            [
                ("foo", uniq_object),
                (("foo", "bar"), uniq_object),
            ],
            prio=123,
        )
    )

    assert Arguments.from_list([]) is Arguments.EMPTY
    assert Arguments.from_list([a := Arguments()]) is a
    assert Arguments.from_list(
        [Fragment.make(("foo",), Value(uniq_object))]
    ) == Arguments([Fragment.make(("foo",), Value(uniq_object))])

    with pytest.raises(TypeError):
        Arguments(123)

    assert repr(Arguments.make({})) == "Arguments()"
    assert (
        repr(Arguments.make(key=123)) == "Arguments(Fragment('key', None, Value(123)),)"
    )
    assert (
        repr(Arguments.make(foo=123, bar=456))
        == "Arguments(Fragment('foo', None, Value(123)), Fragment('bar', None, Value(456)))"
    )

    args = Arguments(
        [
            Fragment.make(("foo",), Value(uniq_object, 1)),
            Fragment.make(("foo", "bar"), Value(uniq_object, 2)),
            Fragment.make(
                ("boo",),
                Arguments(
                    [
                        Fragment.make(("sub",), Value(uniq_object, 3)),
                        Fragment.make(("sub", "bar"), Value(uniq_object, 4)),
                    ]
                ),
            ),
            Arguments(
                [
                    Fragment.make(("nest",), Value(uniq_object, 5)),
                ]
            ),
        ]
    )
    assert list(args.enumerate(nested=False)) == [
        (args[0], args[0].inner),
        (args[1], args[1].inner.inner),
    ]
    assert list(args.enumerate(nested=True)) == [
        (args[0], args[0].inner),
        (args[1], args[1].inner.inner),
        (args[2], args[2].inner[0], args[2].inner[0].inner),
        (args[2], args[2].inner[1], args[2].inner[1].inner.inner),
        (args[3][0], args[3][0].inner),
    ]


class TestParam(TestCase):
    def test_param(self):
        self.assertIs(Param(type=(o := object())).type, o)
        self.assertIs(Param[o := object()]().type, o)
        self.assertRaises(TypeError, lambda: Param[int, str])

        # missing name - not really triggerable by user
        self.assertIs(Param()._name(int), None)

    def test_typed_alias(self):
        with self.assertWarns(TypedAlias.TypedAliasMismatch):

            class WonkyParam[A, B, C](AbstractParam[B]): ...

        WonkyParam[int, str, bool]()

        class Sentinel:
            pass

        class StrangeParam(AbstractParam):
            type = Sentinel

        self.assertIs(StrangeParam().type, Sentinel)

        with self.assertWarns(TypedAlias.TypedAliasRedefined):

            class RedundantParam[T](AbstractParam[T]):
                TypedAlias.register(T, TypedAlias._typevar2name[T])

        with self.assertRaises(TypedAlias.TypedAliasClash):

            class ClashingParam[T](AbstractParam[T]):
                TypedAlias.register(T, "not" + TypedAlias._typevar2name[T])

        with self.assertWarns(TypedAlias.TypedAliasMismatch):

            class MismatchParam[R](AbstractParam[R]): ...

    def test_specialized(self):
        uniq_const = object()
        uniq_aux = object()
        uniq_return = object()
        uniq_override = object()

        class Special(ParaO):
            const = Const(uniq_const)

            prop: object

            @Prop(aux=uniq_aux)
            def prop(self):
                return uniq_return

        with self.assertRaises(AttributeError):
            Special.prop._on_prop_attr
        with self.assertRaises(AttributeError):
            Special.prop.on_func_attr

        Special.prop.func.on_func_attr = attr = object()
        self.assertIs(Special.prop.on_func_attr, attr)

        self.assertIs(Special(const=None).const, uniq_const)
        self.assertIs(Special.prop.aux, uniq_aux)
        self.assertIs(Special().prop, uniq_return)
        self.assertIs(Special(prop=uniq_override).prop, uniq_override)


class TestParaO(TestCase):
    def test_create(self):
        ParaO()

        class Sub(ParaO): ...

        self.assertIsInstance(Sub(), Sub)
        self.assertIsInstance(ParaO({ParaO: Sub}), Sub)
        self.assertIsInstance(ParaO({"__class__": Sub}), Sub)

        # cover some rare branches
        self.assertIsInstance(
            ParaO(
                Arguments(
                    (
                        Arguments.from_dict({ParaO: UNSET}),
                        Arguments.EMPTY,
                        Arguments.from_dict({ParaO: Sub}),
                    )
                )
            ),
            Sub,
        )

        self.assertRaises(TypeError, lambda: ParaO({ParaO: 123}))

        with (
            catch_warnings(action="ignore", category=OwnParameters.CacheReset),
            self.assertRaises(DuplicateParameter),
        ):
            Sub.foo1 = Sub.foo2 = Param()

        self.assertEqual(
            Sub().__repr__(compact="???"),
            "tests.test_core:TestParaO.test_create.<locals>.Sub(???)",
        )

    def test_own_params(self):
        class Sub(ParaO):
            foo: int = Param()
            bar: str = Param()

        self.assertEqual(Sub.__own_parameters__, {"foo": Sub.foo, "bar": Sub.bar})

        with self.assertWarns(OwnParameters.CacheReset):
            Sub.boo = Param(type=float)

        self.assertEqual(Sub.__own_parameters__["boo"], Sub.boo)

        with self.assertWarns(OwnParameters.CacheReset):
            Sub.boo = Param(type=complex)

        self.assertEqual(Sub.__own_parameters__["boo"], Sub.boo)

        with self.assertWarns(OwnParameters.CacheReset):
            del Sub.foo, Sub.bar, Sub.boo

        self.assertEqual(Sub.__own_parameters__, {})

        with self.assertWarns(OwnParameters.CacheReset):
            Sub.boo = None
        del Sub.boo

        Sub.__dunder__ = None
        del Sub.__dunder__

    def test_resolution_simple(self):
        class Sub(ParaO):
            foo: int = Param()
            bar = Param(None, type=str)
            boo = Param[bool]()
            notyp = Param(None)

        self.assertEqual(Sub.boo.type, bool)

        with self.assertRaises(MissingParameterValue):
            Sub().foo
        with self.assertWarns(UntypedParameter):
            Sub().notyp

        self.assertEqual(Sub({"foo": 123}).foo, 123)
        self.assertEqual(Sub({Sub.foo: 123}).foo, 123)
        self.assertEqual(Sub({(Sub, "foo"): 123}).foo, 123)
        self.assertEqual(Sub({(Sub, Sub, "foo"): 123}).foo, 123)
        self.assertEqual(Sub({(Sub, Sub.foo): 123}).foo, 123)

        self.assertEqual(Sub().bar, None)
        self.assertEqual(Sub(bar=123).bar, "123")

        self.assertEqual(Sub({("foo", "N_A"): 123, "N_A": "", "foo": 321}).foo, 321)

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

        obj = Wrap({(Sub, "foo"): 123})
        self.assertEqual(obj.one.foo, 123)
        self.assertEqual(obj.other.foo, 123)

        # late commons
        self.assertEqual(
            Wrap({"foo": 321}, {("one", "N_A"): 123, "foo": 123}).one.foo, 123
        )

    def test_common_base(self):
        class Base(ParaO):
            foo = Param[int](0)

        class Ext1(Base):
            pass

        class Ext2(Base):
            ext1 = Param[Ext1]()

        ext2 = Ext2(foo=1)
        self.assertEqual(ext2.foo, 1)
        self.assertEqual(ext2.ext1.foo, 0)

    def test_non_eager_parameter(self):
        class Foo(ParaO):
            bar = Param[int](eager=False)

        with eager(True):
            foo = Foo()
        with self.assertRaises(MissingParameterValue):
            foo.bar

    def test_expansion(self):
        class Foo(ParaO):
            bar = Param[int]()

        with eager(False):
            f = Foo(bar=[1, 2, 3])
            # raises on access
            self.assertRaises(Expansion, lambda: f.bar)

        with eager(True):
            self.assertRaises(Expansion, lambda: Foo(bar=[1, 2, 3]))
            try:
                Foo(Arguments.from_dict({"unused": 1}), bar=[1, 2, 3])
            except Expansion as exp:
                self.assertEqual(exp.param, Foo.bar)
                self.assertEqual(exp.param_name, "bar")
                self.assertEqual(exp.values, (1, 2, 3))

        self.assertEqual(repr(Expansion([1, 2, 3])), "Expansion(<3 values>)")

    def test_collect(self):
        class Foo(ParaO):
            bar = Param[int]()

        # function based
        func = Mock(return_value=True)

        class Wrap(ParaO):
            foo = Param[Foo](collect=func)

        with eager(True):
            inst = Wrap(bar=[1, 2, 3])
        exp = inst.foo
        func.assert_called_once_with(exp, inst)
        self.assertIsInstance(exp.source, Foo)
        self.assertIsInstance(exp, Expansion)
        # self.assertEqual(exp._unwind, [])
        self.assertEqual(exp.make_key(), ("bar",))
        self.assertEqual(exp.make_key(False), (Foo, "bar"))
        self.assertEqual(exp.make_key(False, use_cls=False), ("bar",))
        self.assertEqual(exp.make_key(False, use_name=False), (Foo, Foo.bar))
        with self.assertWarns(ExpansionGeneratedKeyMissingParameter):
            self.assertEqual(
                exp.make_key(False, use_param=False),
                (Foo, "bar"),
            )
        with self.assertWarns(ExpansionGeneratedKeyMissingParameter):
            self.assertEqual(
                exp.make_key(False, want=(Foo,), use_name=False), (Foo, Foo.bar)
            )
        self.assertEqual(exp.make_key(False, want=(Foo.bar,)), ("bar",))
        self.assertIsInstance(repr(exp), str)

        # bare argument based
        items = [[Foo], [Foo.bar], ["bar"]]
        for coll in items + [[it] for it in items]:
            with self.subTest(coll=coll), eager(True):
                Wrap.foo.collect = coll
                self.assertIsInstance(Wrap(bar=[1, 2, 3]).foo, Expansion)

    def test_expand(self):
        # uses two level expandable scenario

        class Foo(ParaO):
            bar = Param[int]()

        class Mid(ParaO):
            boo = Param[int](0)
            foo = Param[Foo]()

        class Wrap2(ParaO):
            mid = Param[Mid](collect=Mock(return_value=True))

        with eager(True):
            self.assertEqual(Wrap2(bar=[1, 2, 3]).mid.make_key(), ("bar",))
            self.assertEqual(
                Wrap2({("foo", "bar"): [1, 2, 3]}).mid.make_key(), ("foo", "bar")
            )
            self.assertEqual(
                Wrap2(foo=dict(bar=[1, 2, 3])).mid.make_key(), (Mid, "foo", "bar")
            )
            self.assertSequenceEqual(
                list(map(attrgetter("foo.bar"), Wrap2(bar=[1, 2, 3]).mid.expand())),
                [1, 2, 3],
            )
            self.assertSequenceEqual(
                list(
                    map(
                        attrgetter("foo.bar"),
                        Wrap2({("foo", "bar"): [1, 2, 3]}).mid.expand(),
                    )
                ),
                [1, 2, 3],
            )
            self.assertSequenceEqual(
                list(
                    map(
                        attrgetter("foo.bar"),
                        Wrap2({("mid", "foo", "bar"): [1, 2, 3]}).mid.expand(),
                    )
                ),
                [1, 2, 3],
            )
            self.assertSequenceEqual(
                list(
                    map(
                        attrgetter("boo", "foo.bar"),
                        Wrap2(boo=[1, -1], bar=[1, 2, 3]).mid.expand(),
                    )
                ),
                [
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (-1, 1),
                    (-1, 2),
                    (-1, 3),
                ],
            )

    def test_inner(self):
        out1 = Out()
        self.assertEqual(
            tuple(out1.__inner__), (out1.in1, *out1.in2, out1.in3u, out1.in3u)
        )

        out2 = Out(in2=[In(), In()])
        self.assertEqual(
            tuple(out2.__inner__), (out2.in1, *out2.in2, out2.in3u, out2.in3u)
        )

        with eager(True):
            out3 = Out({("in1", "exp"): [1, 2]})
        inner = tuple(out3.__inner__)
        self.assertEqual(inner[0].exp, 1)
        self.assertEqual(inner[1].exp, 2)
        self.assertEqual(inner[2:], (out3.in3u, out3.in3u))

    def test_pickle(self):
        pre = Out({(In, In.exp): 1, "in2": [In(exp=2, uniq=3)], "uniq": -1})
        post = pickle.loads(pickle.dumps(pre))
        self.assertEqual(pre, post)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(bare_param)


class In(ParaO):
    exp = Param[int](0)

    @Prop
    def uniq(self) -> int:
        return id(self)


class Out(ParaO):
    in1 = Param[In](collect=[In.exp])
    in2 = Param[list[In]]([])

    in3u = Param[In](significant=False)

    @Prop
    def in3(self) -> dict:
        return {
            "deep": [
                "nested",
                ("structure", {"with", frozenset({"some", (self.in3u,) * 2})}),
            ]
        }


bare_param = Param[str]()
