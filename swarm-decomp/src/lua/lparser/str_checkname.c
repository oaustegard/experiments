#include "LexState.h"
#include "fn_luaX_next.h"
#include "check.h"

static TString *str_checkname (LexState *ls) {
  TString *ts;
  check(ls, 0x123);
  ts = ls->t.seminfo.ts;
  luaX_next(ls);
  return ts;
}
