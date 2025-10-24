from pprint import PrettyPrinter
from shutil import get_terminal_size

from .core import ParaO


class PPrint(PrettyPrinter):
    def __init__(self, *args, width: int = -1, **kwargs):
        if width < 0:  # pragma: no branch
            width += get_terminal_size().columns
        super().__init__(*args, width=width, **kwargs)

    def pprint(self, object, indent: int = 0, after: str = ""):
        if self._stream is not None:  # pragma: no branch
            assert indent >= 0
            if after:
                after = f": {after}"
            self._stream.write(indent * self._indent_per_level * " ")
            self._format(object, self._stream, indent, len(after), {}, 0)
            self._stream.write(f"{after}\n")

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
