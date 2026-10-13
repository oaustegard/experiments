#include "LexState.h"
#include "fn_luaX_next.h"
#include "check.h"

static void checknext (LexState *ls, int c) {
  check(ls, c);
  luaX_next(ls);
}
