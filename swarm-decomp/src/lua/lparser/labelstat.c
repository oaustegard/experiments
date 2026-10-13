#include "LexState.h"
#include "fn_luaX_next.h"
#include "checkrepeated.h"
#include "createlabel.h"
#include "statement.h"
#include "checknext.h"
#include "block_follow.h"

static void labelstat (LexState *ls, TString *name, int line) {
  checknext(ls, 0x11f);
  while (ls->t.token == ';' || ls->t.token == 0x11f)
    statement(ls);
  checkrepeated(ls, name);
  createlabel(ls, name, line, block_follow(ls, 0));
}
