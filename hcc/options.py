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
    sso_struct: str = "little-endian"  # -fsso-struct=big-endian|little-endian|native
    quote_dirs: list[str] = field(default_factory=list)       # -iquote
    system_dirs: list[str] = field(default_factory=list)      # -isystem
    after_dirs: list[str] = field(default_factory=list)       # -idirafter
    defines: list[tuple[str, str | None]] = field(default_factory=list)  # (-D|-U, text)
    includes: list[str] = field(default_factory=list)         # -include FILE
    std: str = "gnu17"
    nostdinc: bool = False
    trigraphs: bool = False            # -trigraphs (also implied by strict -std=)
    pedantic_errors: bool = False      # -pedantic-errors (pedantic diagnostics become errors)
    gnu89_inline: bool = False         # -fgnu89-inline
    no_asm: bool = False               # -fno-asm (asm/typeof are not keywords)
    plan9_extensions: bool = False     # -fplan9-extensions (`struct S;` member is anonymous)
    short_wchar: bool = False          # -fshort-wchar (wchar_t is unsigned short, L"" units are 16-bit)
    char_unsigned: bool = False        # -funsigned-char (plain char is unsigned; only checked by constexpr)
    wrapv: bool = False                # -fwrapv (signed overflow wraps; default -fno-wrapv: undefined)
    permissive: bool = False           # -fpermissive (int-conversion errors become warnings)
    linker_args: list[str] = field(default_factory=list)
    keep_temps: bool = False
    dump_ast: bool = False
    dump_tokens: bool = False
    pic: bool = True                  # __PIC__/__pic__ (gcc on x86-64 defaults to -fPIE)
    pie: bool = True                  # __PIE__/__pie__ (-fPIE, the default; -fPIC clears it)
    language: str | None = None        # -x c / -x assembler
