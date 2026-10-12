"""Options: everything parsed from the command line (one object per run).

Created by hcc.driver.parse_args(); passed read-only to every stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Options:
    inputs: list[str] = field(default_factory=list)
    output: str | None = None
    mode: str = "link"                 # "preprocess" | "asm" | "obj" | "link"
    include_dirs: list[str] = field(default_factory=list)     # -I
    quote_dirs: list[str] = field(default_factory=list)       # -iquote
    system_dirs: list[str] = field(default_factory=list)      # -isystem
    after_dirs: list[str] = field(default_factory=list)       # -idirafter
    defines: list[tuple[str, str | None]] = field(default_factory=list)  # (-D|-U, text)
    includes: list[str] = field(default_factory=list)         # -include FILE
    std: str = "gnu11"
    nostdinc: bool = False
    linker_args: list[str] = field(default_factory=list)
    keep_temps: bool = False
    dump_ast: bool = False
    dump_tokens: bool = False
    pic: bool = False
    language: str | None = None        # -x c / -x assembler
