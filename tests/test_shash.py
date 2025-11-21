import sys
from collections import OrderedDict
from types import ModuleType
from unittest import TestCase
from unittest.mock import Mock, patch

from parao.core import Param, ParaO
from parao.shash import (
    UnsupportedError,
    UsesLocalsError,
    UsesMainWarning,
    UsesPickleWarning,
    _SHash,
    bin_hash,
    hex_hash,
)


def fake_main_func(): ...


fake_main_func.__module__ = "__main__"


class Unpickable:
    __reduce__ = None


class ndarray:
    def tolist(self): ...


class custom_int(int): ...


class Custom(ParaO):
    foo = Param[int](1)
    bar = Param[str](neutral="old")
    ignored = Param(significant=False)


class TestShash(TestCase):
    def test_values(self):
        list_in_self1 = []
        list_in_self1.append(list_in_self1)
        list_in_self2 = []
        list_in_self2.append([list_in_self2])

        # fmt: off
        self.assertEqual(hex_hash(1), "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b")
        self.assertEqual(hex_hash(1.234), "00a4e3d579394c0826980361ff3ca00e4915d0dc33b3bcb8775453aed9e8b16d")
        self.assertEqual(hex_hash(5.678j), "2852241032ed1e4fba97014e64f3e1935f2a9eaa2093755590f851583bfb0377")
        self.assertEqual(hex_hash("foo"), "af693da41462ec24753ce51403bd1ee5b500cd6941f62f3c97456dd8ffa7f952")
        self.assertEqual(hex_hash(b"bar"), "0f0cd6bc85f1855d43efd8f861ecd52a7a9360b7cd32537c85118e24d657abae")
        self.assertEqual(hex_hash(None), "dc937b59892604f5a86ac96936cd7ff09e25f18ae6b758e8014a24c7fa039e91")
        self.assertEqual(hex_hash(Ellipsis), "4637e99b28a1ce112e8f4009dbf144f3a75a04d3e4b0c30b084aef21c5e3cfe6")
        self.assertEqual(hex_hash((1, 2, 3)), "42f483f79545c1746890398387b591f019c11d69c6d86c2aad1375c70ce63b43")
        self.assertEqual(hex_hash([1, 2, 3]), "7a65d251c6a3708230a39017f4ee7c7bf5036abe9002093d2ea337593dffa164")
        self.assertEqual(hex_hash({1, 2, 3}), "058353b989d130371d502fd7730261695cd2e76e884e415ef00cbfb147d833c7")
        self.assertEqual(hex_hash(frozenset({1, 2, 3})), "7f1cc479a8fa9c9ab41aaaa0804169febd62d7ee7ee674a29b87076efbe6e9f0")
        self.assertEqual(hex_hash({1: 2, 3: 4}), "4b2b7e92da0e61c046a3d0a8626342bdb415b7acac138d33c18bb516877e1d66")
        self.assertEqual(hex_hash(OrderedDict({1: 2, 3: 4})), "3d9cd99e702a126ca217205648b564959805edc71fe20a4a4d78abe265d2321a")
        self.assertEqual(hex_hash(OrderedDict({3: 4, 1: 2})), "56305e4941ba58d828f7e406ea94c527ddb859c3a01caad9bed385d2dd0de944")
        self.assertEqual(hex_hash(range(1, 2, 3)), "1cc18cce33f088254d3a7f0d04ef38f06de245547c27f19e8b51647f90b9a520")
        self.assertEqual(hex_hash(slice(1, 2, 3)), "5a6c5731911d7c5bdedf4d9745d43ee2607a372650987c0448f273f087fd2be8")
        # wonky types
        self.assertEqual(hex_hash(sys), "518b67e652531c5fe7e25d6b2c3b4ef6224e7d90da2091967dd47eb082b26a19")
        self.assertEqual(hex_hash(list_in_self1), "16db3cdec82fa1efb8cd6bdcaf25bbc9cf35e5f9a11444d61ddc924b8f45df2d")
        self.assertEqual(hex_hash(list_in_self2), "28f0ac587249fed6eae6d8e29cb25fec492f92979a52362b06fcddfdbd44f61e")
        # fmt: on

    def test_equivalence(self):
        self.assertEqual(bin_hash(None).hex(), hex_hash(None))

        self.assertEqual(hex_hash(set(range(100))), hex_hash(set(reversed(range(100)))))

        self.assertEqual(hex_hash({1: 2, 3: 4}), hex_hash({3: 4, 1: 2}))

        self.assertEqual(hex_hash(custom_int(5)), hex_hash(5))

        self.assertNotEqual(hex_hash([list, []]), hex_hash([[list]]))

    def test_difficulties(self):
        self.assertRaises(UnsupportedError, lambda: bin_hash(Unpickable()))

    def test_bad_unpickable(self):
        for probe in [
            object(),
            # from types
            len,
            [].append,
            object.__init__,
            object().__str__,
            str.join,
        ]:
            with self.subTest(probe=probe), self.assertWarns(UsesPickleWarning):
                bin_hash(probe)

    def test_locals(self):
        def make_closure():
            a = 1

            def func():
                return a  # pragma: no cover

            return func

        self.assertRaises(UsesLocalsError, lambda: bin_hash(make_closure()))

    def test_main(self):
        self.assertWarns(UsesMainWarning, lambda: bin_hash(fake_main_func))

    def test_numpy(self):
        numpy = Mock(spec=ModuleType)

        numpy.ndarray = ndarray
        numpy.dtype = Mock(spec=type)

        with patch.dict(sys.modules, numpy=numpy):
            del sys.modules["parao.shash"]

            from parao.shash import hex_hash

            nda = Mock(spec=ndarray)
            nda.tolist = Mock(return_value=[[[1], [2], [3]]])

            self.assertEqual(
                hex_hash(nda),
                "38cbdfe4ca762f8eecf90ad461afe5b440ef1a89f525b87c450776ace64e5367",
            )

            nda.tolist.assert_called_once()

    def test_custom(self):
        custom = Mock()
        custom.__shash__ = Mock()
        custom.__shash__.return_value = b"foobar"

        s = _SHash()

        self.assertEqual(s(custom), b"foobar")
        custom.__shash__.assert_called_once_with(s)

    def test_parao(self):
        c = Custom(bar="")
        self.assertIs(bin_hash(c), bin_hash(c))
        self.assertEqual(repr(c), "tests.test_shash:Custom(bar='')")

        self.assertEqual(
            hex_hash(Custom(foo=1, bar="val")),
            "e9259685825ceb325388dbc04057aa20c3abe7888cff403a63fb8ae92027a466",
        )
        self.assertEqual(
            hex_hash(Custom(bar="val")),
            "e9259685825ceb325388dbc04057aa20c3abe7888cff403a63fb8ae92027a466",
        )
        self.assertEqual(
            hex_hash(Custom(bar="val", foo=2)),
            "ef3ba52e2d4ddbe698e1adb3b8451bc8cf1e2157f63105af0c8ee3229c5f2ecb",
        )

        with patch.object(Custom.bar, "neutral", "old"):
            hash_old = hex_hash(Custom(bar="old"))
        with patch.object(Custom.bar, "significant", False):
            hash_ign = hex_hash(Custom())
        self.assertEqual(hash_old, hash_ign)
