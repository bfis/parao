from typing import Any
from unittest import TestCase
from unittest.mock import ANY, Mock, call

from parao.action import (
    Action,
    MissingParameterValueOrOverride,
    Plan,
    RecursiveAction,
    ValueAction,
)
from parao.core import ParaO, Param, UntypedParameter


class TestAction(TestCase):
    def test_action(self):
        func = Mock()

        class Foo(ParaO):
            act = Action[Any, None, []](func)

        sentinel = object()

        with Plan().use(run=True):
            foo = Foo({"act": sentinel})
        func.assert_called_once_with(foo, sentinel)

        func.reset_mock()
        foo = Foo()
        foo.act()
        func.assert_called_once_with(foo)

    def test_value_action(self):
        mock = Mock()

        class Foo(ParaO):
            @ValueAction
            def act(self, value: int):
                return mock(self, value)

            @ValueAction
            def act2(self, value):
                return (123, value)

        with self.assertRaises(MissingParameterValueOrOverride):
            Foo().act()

        with Plan().use(run=True), self.assertWarns(UntypedParameter):
            foo = Foo({"act": 123})
        mock.assert_called_once_with(foo, 123)

        mock.reset_mock()
        foo.act()
        mock.assert_called_once_with(foo, 123)

        mock.reset_mock()
        mock.return_value = 456
        self.assertEqual(foo.act(321), 456)
        mock.assert_called_once_with(foo, 321)

        sentinel = object()
        self.assertEqual(foo.act2(sentinel), (123, sentinel))

    def test_recursive_action(self):
        mock = Mock(return_value=None)

        class FooBase(ParaO):
            act = RecursiveAction(mock)

        class Foo1(FooBase):
            pass

        class Foo2(FooBase):
            foo1 = Param[Foo1]()

        class Foo3(FooBase):
            foo2 = Param[Foo2]()

        with Plan().use(run=True):
            foo3 = Foo3(act=True)
        mock.assert_has_calls(
            [call(foo3, 0, ANY), call(foo3.foo2, 1, ANY), call(foo3.foo2.foo1, 2, ANY)]
        )

        mock.reset_mock()
        foo3.act(1)
        mock.assert_has_calls([call(foo3, 0, ANY), call(foo3.foo2, 1, ())])

        mock.reset_mock()
        (foo3 := Foo3()).act()
        mock.assert_has_calls(
            [call(foo3, 0, ANY), call(foo3.foo2, 1, ANY), call(foo3.foo2.foo1, 2, ANY)]
        )

        mock.reset_mock()
        (foo3 := Foo3({(Foo1, "act"): False})).act()
        mock.assert_has_calls([call(foo3, 0, ANY), call(foo3.foo2, 1, ANY)])

        mock.reset_mock()
        (foo3 := Foo3()).act(1)
        mock.assert_has_calls([call(foo3, 0, ANY), call(foo3.foo2, 1, ())])

        mock.reset_mock()
        (foo3 := Foo3()).act(0)
        mock.assert_has_calls([call(foo3, 0, ())])

        # self.assertEqual(foo3.act, ())
