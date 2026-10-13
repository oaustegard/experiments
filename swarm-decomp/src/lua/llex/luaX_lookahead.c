#include "llex_macros.h"
#include "llex.h"
#include "fn_luaX_lookahead.h"

int luaX_lookahead(LexState *ls) {
  ls->lookahead.token = llex(ls, &ls->lookahead.seminfo);
  return ls->lookahead.token;
}
