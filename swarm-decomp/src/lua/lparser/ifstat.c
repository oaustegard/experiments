#include "LexState.h"
#include "FuncState.h"
#include "fn_luaK_patchtohere.h"
#include "fn_luaX_next.h"
#include "check_match.h"
#include "block.h"
#include "test_then_block.h"

static void ifstat (LexState *ls, int line) {
  FuncState *fs = ls->fs;
  int escapelist = -1;
  test_then_block(ls, &escapelist);
  while (ls->t.token == 0x104)
    test_then_block(ls, &escapelist);
  if (ls->t.token == 0x103) {
    luaX_next(ls);
    block(ls);
  }
  check_match(ls, 0x105, 0x10a, line);
  luaK_patchtohere(fs, escapelist);
}
