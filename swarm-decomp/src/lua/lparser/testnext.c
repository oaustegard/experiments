#include "LexState.h"
#include "fn_luaX_next.h"

static int testnext (LexState *ls, int c) {
  if (ls->t.token == c) {
    luaX_next(ls);
    return 1;
  }
  return 0;
}
