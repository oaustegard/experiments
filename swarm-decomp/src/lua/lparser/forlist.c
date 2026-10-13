#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "fn_luaK_checkstack.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaX_newstring.h"
#include "fn_luaX_next.h"
#include "checknext.h"
#include "str_checkname.h"
#include "new_localvar.h"
#include "adjustlocalvars.h"
#include "adjust_assign.h"
#include "explist.h"
#include "forbody.h"
#include "subexpr.h"

static void forlist (LexState *ls, TString *indexname) {
  FuncState *fs = ls->fs;
  expdesc e;
  int nvars = 5;
  int line;
  int base = fs->freereg;
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, indexname);
  while (ls->t.token == ',') {
    luaX_next(ls);
    new_localvar(ls, str_checkname(ls));
    nvars++;
  }
  checknext(ls, 0x10b);
  line = ls->linenumber;
  adjust_assign(ls, 4, explist(ls, &e), &e);
  adjustlocalvars(ls, 4);
  ls->fs->bl->upval = 1;
  ls->fs->bl->insidetbc = 1;
  fs->needclose = 1;
  luaK_checkstack(fs, 3);
  forbody(ls, base, line, nvars - 4, 1);
}
