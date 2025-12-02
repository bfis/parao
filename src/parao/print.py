from io import StringIO
from pprint import PrettyPrinter
from shutil import get_terminal_size

from .core import ParaO


class Wood:
    'The "wood" the "branches" are made out of, to hold some "leaves".'

    __slots__ = ("line_1", "line_x")
    line_1: tuple[str, str]
    line_x: tuple[str, str]

    def __init__(self, d: str, f: str, r: str, h: str, e: str, pad_to: int = 0):
        self.line_1 = r.ljust(pad_to, h), f.ljust(pad_to, h)
        self.line_x = e.ljust(pad_to, e), d.ljust(pad_to, e)

    def __call__(self, more: int, depth: int = 0):
        depth = max(depth, more.bit_length())
        if not depth:
            return "", "", ""
        *more, last = [b == "1" for b in bin(more)[2:].rjust(depth, "0")]

        com = "".join(self.line_x[m] for m in more)
        ref = self.line_x[0] * depth

        return com + self.line_1[last], com + self.line_x[last], ref


class PPrint(PrettyPrinter):
    _wood_utf = Wood(*"│├└─ ", 2)
    _wood_alt = Wood(*"|++- ", 2)

    def __init__(self, *args, width: int = -1, wood: Wood = None, **kwargs):
        if width < 0:  # pragma: no branch
            width += get_terminal_size().columns
        super().__init__(*args, width=width, **kwargs)
        if wood is None:
            if getattr(self._stream, "encoding", "").lower().startswith("utf"):
                wood = self._wood_utf
            else:
                wood = self._wood_alt
        self.wood = wood

    def pleaf(self, object, depth: int, more: int, after: str = ""):
        'Print a "leaf" object, supported by some "branches" made out of "wood".'
        first, rest, ref = self.wood(more, depth)

        if after:
            after = f": {after}"

        sio = StringIO()
        sio.write(first)
        self._format(object, sio, len(ref), len(after), {}, 0)
        sio.write(f"{after}\n")

        self._stream.write(sio.getvalue().replace(f"\n{ref}", f"\n{rest}"))

    def _pprint_parao(self, object: ParaO, stream, indent, allowance, context, level):
        pre = f"{object.__class__.__fullname__}("
        stream.write(pre)

        if level > 1:
            stream.write("...")
        else:
            items = [
                (name, value)
                for name, value, neutral in object.__rich_repr__()
                if value != neutral
            ]
            if self._sort_dicts:  # pragma: no branch
                items.sort()
            stream.write("\n " + " " * indent)
            self._format_namespace_items(
                items, stream, indent + 1, allowance + 1, context, level
            )
            stream.write("\n" + " " * indent)

        stream.write(")")

    _dispatch = PrettyPrinter._dispatch.copy()
    _dispatch[ParaO.__repr__] = _pprint_parao
