#include "llex_macros.h"
#include "save.h"
#include "lexerror.h"

static void inclinenumber(LexState *ls) {
  int old = ls->current;
  next(ls);
  if ((ls->current == '\n' || ls->current == '\r') && ls->current != old)
    next(ls);
  if (++ls->linenumber >= 0x7fffffff)
    lexerror(ls, "chunk has too many lines", 0);
}
