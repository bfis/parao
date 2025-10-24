from typing import Type

from .action import RecursiveAction
from .core import AbstractParam, Const, ParaO
from .output import Output
from .print import PPrint
from .run import RunAction

pprint = PPrint()


class Task[R](ParaO):
    code_version: Const
    run: RunAction[R]
    output: Type[Output] = Output

    def __init_subclass__(cls):
        v = cls.__dict__.get("code_version")
        if v is not None and not isinstance(v, AbstractParam):
            cls.code_version = Const(v)
        r = cls.__dict__.get("run")
        if r is not None and not isinstance(r, AbstractParam):
            cls.run = RunAction(r)
        return super().__init_subclass__()

    @RecursiveAction
    def remove(self, depth: int):
        out = self.run.output
        if out.exists:
            out.remove()
            after = "removed"
        else:
            after = "missing"
        pprint.pprint(self, indent=2 * depth, after=after)

    @RecursiveAction
    def status(self, depth: int):
        after = "done" if self.run.done else "missing"
        pprint.pprint(self, indent=2 * depth, after=after)

    @RecursiveAction
    def print(self, depth: int):
        pprint.pprint(self, indent=2 * depth)
