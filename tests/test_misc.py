from unittest import TestCase
from parao.misc import ContextValue, safe_len, safe_repr


class TestContextValue(TestCase):
    def test_defaults(self):

        cv = ContextValue("cv")
        sentinel = object()
        self.assertIs(cv(default=sentinel), sentinel)


class TestMisc(TestCase):
    def test_safe(self):

        class Foo:
            def __repr__(self):
                raise RuntimeError()

        self.assertRaises(RuntimeError, lambda: repr(Foo()))
        self.assertEqual(safe_repr(o := Foo()), object.__repr__(o))

        self.assertRaises(TypeError, lambda: len(Foo()))
        self.assertEqual(safe_len(Foo(), (o := object())), o)
