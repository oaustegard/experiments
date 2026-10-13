#pragma once
/* counters and loop labels of the statement reader (zero-initialised) */
static int enums_n, funcs_n, l_cont, l_break;
/* depth of the ?: expression reader and the case expression (zero-initialised) */
static int ncexpr, caseexpr;
