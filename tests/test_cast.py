from typing import Annotated, Any
from unittest import TestCase
from parao.cast import cast


class TestCasting(TestCase):
    def test_primitives(self):
        self.assertEqual(cast("123", int), 123)
        self.assertEqual(cast("1.3", float), 1.3)
        self.assertEqual(cast("1j", complex), 1j)
        self.assertEqual(cast(123, str), "123")

        # None
        self.assertEqual(cast(None, None), None)
        self.assertEqual(cast(None, type(None)), None)
        self.assertRaises(ValueError, lambda: cast(True, None))

    def test_containers(self):

        self.assertEqual(cast(["123"], list[int]), [123])
        self.assertEqual(cast({"123"}, set[int]), {123})
        self.assertEqual(cast(frozenset({"123"}), frozenset[int]), frozenset({123}))

        self.assertEqual(cast({"123": 456}, dict[int, str]), {123: "456"})
        self.assertEqual(cast([("123", 456)], dict[int, str]), {123: "456"})

        # empty tuple
        self.assertEqual(cast([], tuple[()]), ())
        self.assertRaises(ValueError, lambda: cast([1], tuple[()]))

        # any tuple
        self.assertEqual(cast([1, 2, 3], tuple), (1, 2, 3))
        self.assertEqual(cast([1, 2, 3], tuple[Any, ...]), (1, 2, 3))
        self.assertRaises(TypeError, lambda: cast([1], tuple[...]))  # type: ignore

        # fixed tuple
        self.assertEqual(cast([1, 2, 3], tuple[int, str, float]), (1, "2", 3.0))
        self.assertRaises(ValueError, lambda: cast([], tuple[int]))

    def test_complex(self):
        self.assertRaises(ValueError, lambda: cast(1.2, int))
        self.assertEqual(cast("1.2", int | float), 1.2)
        self.assertRaises(TypeError, lambda: cast("foo", int | float))
        self.assertEqual(cast("123", Annotated[int, str]), 123)

        class Foo:
            @classmethod
            def __cast_from__(cls, value, original_type):
                raise NotImplementedError

        self.assertRaises(TypeError, lambda: cast(1, Foo))
