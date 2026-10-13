#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "ConsControl.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_code.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaK_settablesize.h"
#include "fn_luaX_next.h"
#include "fn_luaX_lookahead.h"
#include "error_expected.h"
#include "check_match.h"
#include "subexpr.h"
#include "recfield.h"
#include "closelistfield.h"
#include "lastlistfield.h"
#include "expkind.h"
#include "testnext.h"
#include "init_exp.h"
#include "field.h"

static void constructor (LexState *ls, expdesc *t) {
  FuncState *fs = ls->fs;
  int line = ls->linenumber;
  int pc = luaK_codeABCk(fs, 0x13, 0, 0, 0, 0);
  ConsControl cc;
  luaK_code(fs, 0);
  cc.na = cc.nh = cc.tostore = 0;
  cc.t = t;
  init_exp(t, VNONRELOC, fs->freereg);
  luaK_reserveregs(fs, 1);
  init_exp(&cc.v, VVOID, 0);
  if (ls->t.token != '{')
    error_expected(ls, '{');
  luaX_next(ls);
  do {
    if (ls->t.token == '}')
      break;
    closelistfield(fs, &cc);
    field(ls, &cc);
  } while (testnext(ls, ',') || testnext(ls, ';'));
  check_match(ls, '}', '{', line);
  lastlistfield(fs, &cc);
  luaK_settablesize(fs, pc, t->u.info, cc.na, cc.nh);
}
