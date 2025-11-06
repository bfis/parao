from unittest import TestCase
from parao.misc import ContextValue, PeekableIter, safe_len, safe_repr, is_subseq


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

    def test_peekable(self):
        tpl = object(), object(), object()

        pi = PeekableIter(tpl)

        self.assertIs(pi.peek(), tpl[0])
        self.assertIs(pi.peek(), tpl[0])
        self.assertEqual(tuple(pi), tpl)
        self.assertRaises(StopIteration, lambda: pi.peek())
        self.assertIs(pi.peek(o := object()), o)

    def test_is_subseq(self):
        self.assertTrue(is_subseq("india", "indonesia"))
        self.assertTrue(is_subseq("oman", "romania"))
        self.assertTrue(is_subseq("mali", "malawi"))
        self.assertFalse(is_subseq("mali", "banana"))
        self.assertFalse(is_subseq("ais", "indonesia"))
        self.assertFalse(is_subseq("ca", "abc"))
