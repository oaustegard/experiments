#include "llex_macros.h"
#include "save.h"
#include "lexerror.h"
#include "skip_sep.h"
#include "inclinenumber.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaX_newstring.h"
#include "fn_luaZ_fill.h"

static void read_long_string(LexState *ls, SemInfo *seminfo, size_t sep) {
  int line = ls->linenumber;
  save(ls, ls->current);
  next(ls);
  if (ls->current == '\n' || ls->current == '\r')
    inclinenumber(ls);
  for (;;) {
    switch (ls->current) {
    case -1: {
      const char *what = (seminfo ? "string" : "comment");
      const char *msg = luaO_pushfstring(ls->L, "unfinished long %s (starting at line %d)", what, line);
      lexerror(ls, msg, 0x120);
      break;
    }
    case ']': {
      if (skip_sep(ls) == sep) {
        save(ls, ls->current);
        next(ls);
        goto endloop;
      }
      break;
    }
    case '\n':
    case '\r': {
      save(ls, '\n');
      inclinenumber(ls);
      if (!seminfo)
        ls->buff->n = 0;
      break;
    }
    default: {
      if (seminfo) {
        save(ls, ls->current);
        next(ls);
      } else
        next(ls);
    }
    }
  }
endloop:
  if (seminfo)
    seminfo->ts = luaX_newstring(ls, ls->buff->buffer + sep, ls->buff->n - 2 * sep);
}
