from typing import Any
from unittest import TestCase
from unittest.mock import Mock, call, patch

from parao.action import Plan, RecursiveAction, SimpleAction, ValueAction
from parao.core import Param, ParaO, Prop, UntypedParameter, Value


class FooBase(ParaO):
    act = RecursiveAction(lambda: None)
    aux = Param[int](0)


class FooNoAct(FooBase):
    act = None


class FooOtherAct(FooBase):
    act = Param[Any](None)


class Foo1(FooBase):
    foo_no_act = Param[FooNoAct]()
    foo_other_act = Param[FooOtherAct]()


class Foo2(FooBase):
    foo1 = Param[Foo1]()


class Foo3(FooBase):
    foo2 = Param[Foo2]()


class FooMix(FooBase):
    @Prop
    def inner(self) -> list[FooBase]:
        return [Foo1(), Foo2()][::-1]

    @SimpleAction
    def interrupter(self): ...


class FooBad(FooBase):
    @Prop
    def bad(self) -> Foo2:
        Foo1(act=True, aux=1).act  # dead "branch"
        return Foo2()


class TestAction(TestCase):
    def test_simple(self):
        func = Mock()

        class Foo(ParaO):
            act = SimpleAction(func)

        Plan().run([Foo()])
        func.assert_not_called()

        Plan().run([Foo({"act": False})])
        func.assert_not_called()

        foo = Plan().run1(Foo({"act": True}))
        func.assert_called_once_with(foo)

        func.reset_mock()
        foo = Foo()
        foo.act()
        func.assert_called_once_with(foo)

    def test_cov_method_1st_arg_annotation(self):
        class Bad1(ParaO):
            @ValueAction
            def bad_signature1(self): ...

        with self.assertRaises(TypeError), self.assertWarns(UntypedParameter):
            Bad1(bad_signature1=1).bad_signature1()

        class Bad2(ParaO):
            @ValueAction
            def bad_signature2(self, *, kw1, kw2): ...

        with self.assertRaises(TypeError), self.assertWarns(UntypedParameter):
            Bad2(bad_signature2=2).bad_signature2()

    def test_value_action(self):
        mock = Mock()

        class Foo(ParaO):
            @ValueAction
            def act(self, value: int):
                return mock(self, value)

            @ValueAction
            def act_no_type(self, value):
                return (123, value)

            @ValueAction[int, None]
            def act2(self, value): ...

        with self.assertRaises(TypeError):
            Foo().act()

        with self.assertWarns(UntypedParameter):
            Plan().run([foo := Foo({"act": 123})])
        mock.assert_called_once_with(foo, 123)

        mock.reset_mock()
        foo.act()
        mock.assert_called_once_with(foo, 123)

        mock.reset_mock()
        mock.return_value = 456
        self.assertEqual(foo.act(321), 456)
        mock.assert_called_once_with(foo, 321)

        sentinel = object()
        self.assertEqual(foo.act_no_type(sentinel), (123, sentinel))

    def test_recursive_action(self):
        mock = Mock(return_value=None)

        with patch.object(FooBase.act, "func", mock):
            mock.reset_mock()
            mix4 = Plan.run1(FooMix({"act": 3, ("inner", "act"): True}))
            mock.assert_has_calls(
                [
                    call(mix4, 0),
                    call(mix4.inner[0], 1),
                    call(mix4.inner[0].foo1, 2),
                    call(mix4.inner[1], 1),
                ]
            )
            self.assertEqual(mock.call_count, 4)

            mock.reset_mock()
            mix6 = Plan.run1(
                FooMix(
                    {
                        "act": 3,
                        "interrupter": Value(True, position=1),
                        (Foo2, "act"): Value(True, position=2),
                    }
                )
            )
            mock.assert_has_calls(
                [
                    call(mix6, 0),
                    call(mix6.inner[0], 1),
                    call(mix6.inner[0].foo1, 2),
                    call(mix6.inner[1], 1),
                    call(mix6.inner[0], 0),
                    call(mix6.inner[0].foo1, 1),
                ]
            )
            self.assertEqual(mock.call_count, 6)

            mock.reset_mock()
            mixB = Plan.run1(FooBad(act=True))
            mock.assert_has_calls(
                [
                    call(mixB, 0),
                    call(mixB.bad, 1),
                    call(mixB.bad.foo1, 2),
                ]
            )
            self.assertEqual(mock.call_count, 3)

            foo3 = Plan.run1(Foo3(act=True))
            mock.assert_has_calls(
                [call(foo3, 0), call(foo3.foo2, 1), call(foo3.foo2.foo1, 2)]
            )

            mock.reset_mock()
            foo3.act(1)
            mock.assert_has_calls([call(foo3, 0), call(foo3.foo2, 1)])

            mock.reset_mock()
            (foo3 := Foo3()).act()
            mock.assert_has_calls(
                [call(foo3, 0), call(foo3.foo2, 1), call(foo3.foo2.foo1, 2)]
            )

            mock.reset_mock()
            (foo3 := Foo3({(Foo1, "act"): False})).act()
            mock.assert_has_calls([call(foo3, 0), call(foo3.foo2, 1)])

            mock.reset_mock()
            (foo3 := Foo3()).act(1)
            mock.assert_has_calls([call(foo3, 0), call(foo3.foo2, 1)])

            mock.reset_mock()
            (foo3 := Foo3()).act(0)
            mock.assert_has_calls([call(foo3, 0)])

            mock.reset_mock()
            mock.return_value = True
            (foo3 := Foo3()).act()
            mock.assert_has_calls([call(foo3, 0)])
