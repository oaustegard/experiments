"""Self-registration for hcc's extension points (CONTRACT section 0, rule 7).

Adding a feature must mean adding a file, not appending to a shared list: two
workers that add unrelated builtins then never touch the same lines. Two tools:

* Registry: a name -> handler table filled by decorators. The modules of its
  packages are imported (sorted by name) on the first lookup, so a new module
  in such a package is picked up with no import list to edit:

      @builtin_parser("__builtin_foo")      # hcc/builtins/foo.py
      def _parse_foo(self, tok): ...

  Registering a name twice raises (both places are named), so the import order
  never decides which handler wins.

* assemble_mixins(package, base): the bases of a class built from mixins
  (Parser, CodeGen, Preprocessor). Every class whose name ends in `Mixin` and is
  defined at top level of a module of `package` is a base; `base` comes last.
  An attribute defined by two of them raises, so the MRO order never matters.
"""
from __future__ import annotations

import importlib
import pkgutil


def import_all(package: str) -> list:
    """Import every module (not subpackage) of `package`, sorted by name; return them.
    Modules whose name starts with '_' are skipped."""
    pkg = importlib.import_module(package)
    mods = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if info.ispkg or info.name.startswith("_"):
            continue
        mods.append(importlib.import_module(f"{package}.{info.name}"))
    return mods


def _where(fn) -> str:
    code = getattr(fn, "__code__", None)
    if code is None:
        return repr(fn)
    return f"{fn.__module__}.{fn.__qualname__} ({code.co_filename}:{code.co_firstlineno})"


class Registry:
    """name -> handler, filled by `@registry(name, ...)` in the modules of `packages`."""

    def __init__(self, what: str, *packages: str):
        self.what = what
        self.packages = packages
        self._table: dict = {}
        self._loaded = False

    def __call__(self, *names: str):
        """Decorator: register the function under each of `names`."""
        def deco(fn):
            for name in names:
                self.add(name, fn)
            return fn
        return deco

    def add(self, name: str, fn) -> None:
        old = self._table.get(name)
        if old is not None and old is not fn:
            raise RuntimeError(f"{self.what} '{name}' registered twice: {_where(old)} and {_where(fn)}")
        self._table[name] = fn

    def load(self) -> "Registry":
        if not self._loaded:
            self._loaded = True
            for p in self.packages:
                import_all(p)
        return self

    def get(self, name: str, default=None):
        if not self._loaded:
            self.load()
        return self._table.get(name, default)

    def __contains__(self, name) -> bool:
        if not self._loaded:
            self.load()
        return name in self._table

    def __iter__(self):
        return iter(self.load()._table)

    def __len__(self) -> int:
        return len(self.load()._table)


def assemble_mixins(package: str, base: type) -> tuple:
    """Bases for a mixin-assembled class: every `*Mixin` class defined in a module of
    `package` (sorted by module, then name), then `base`."""
    mixins = []
    for mod in import_all(package):
        for name, obj in sorted(vars(mod).items()):
            if isinstance(obj, type) and name.endswith("Mixin") and obj.__module__ == mod.__name__:
                mixins.append(obj)
    owner: dict = {}
    for cls in mixins + [base]:
        for attr in vars(cls):
            if attr.startswith("__") and attr.endswith("__"):
                continue
            if attr in owner:
                raise TypeError(f"'{attr}' is defined by both {owner[attr].__module__}.{owner[attr].__name__} "
                                f"and {cls.__module__}.{cls.__name__}: rename one (mixins must not override)")
            owner[attr] = cls
    return tuple(mixins) + (base,)
