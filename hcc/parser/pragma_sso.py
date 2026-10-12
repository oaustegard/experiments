"""#pragma scalar_storage_order state, applied to struct/union definitions that follow.

The preprocessor forwards each `#pragma scalar_storage_order X` as a marker token
whose value is the argument texts. `parser.sso_reverse` is True after
`big-endian`, False after `little-endian` or `default` (the -fsso-struct default,
which is little-endian on x86-64).
"""
from __future__ import annotations

from hcc import diag
from hcc.preprocess.directives import PRAGMA_SSO_MARK


def at_sso_marker(parser) -> bool:
    return parser.at(PRAGMA_SSO_MARK)


def consume_sso_marker(parser) -> None:
    tok = parser.next()
    arg = "".join(a for a in (tok.value or []) if a not in ("(", ")"))   # "big" "-" "endian"
    if arg == "big-endian":
        parser.sso_reverse = True
    elif arg == "little-endian":
        parser.sso_reverse = False
    elif arg == "default":
        parser.sso_reverse = parser.sso_default
    else:
        diag.error(tok, "unknown argument to '#pragma scalar_storage_order'")
