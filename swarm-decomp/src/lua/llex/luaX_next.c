#include "llex_macros.h"
#include "llex.h"
#include "fn_luaX_next.h"

void luaX_next(LexState *ls) {
  ls->lastline = ls->linenumber;
  if (ls->lookahead.token != 0x120) {
    ls->t = ls->lookahead;
    ls->lookahead.token = 0x120;
  } else
    ls->t.token = llex(ls, &ls->t.seminfo);
}
