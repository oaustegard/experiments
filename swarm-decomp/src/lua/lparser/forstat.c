#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"
#include "check_match.h"
#include "enterblock.h"
#include "leaveblock.h"
#include "str_checkname.h"
#include "fornum.h"
#include "forlist.h"

static void forstat (LexState *ls, int line) {
  FuncState *fs = ls->fs;
  TString *varname;
  BlockCnt bl;
  enterblock(fs, &bl, 1);
  luaX_next(ls);
  varname = str_checkname(ls);
  switch (ls->t.token) {
    case '=': fornum(ls, varname, line); break;
    case ',': case 0x10b: forlist(ls, varname); break;
    default: luaX_syntaxerror(ls, "'=' or 'in' expected");
  }
  check_match(ls, 0x105, 0x107, line);
  leaveblock(fs);
}
