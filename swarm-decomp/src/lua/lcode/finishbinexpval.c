#include "FuncState.h"
#include "expdesc.h"
#include "expkind.h"
#include "OpCode.h"
#include "TMS.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_fixline.h"
#include "fn_luaK_code.h"
#include "freeexps.h"

static void finishbinexpval (FuncState *fs, expdesc *e1, expdesc *e2, OpCode op, int v2, int flip, int line, OpCode mmop, TMS event) {
  int v1 = luaK_exp2anyreg(fs, e1);
  int pc = luaK_codeABCk(fs, op, 0, v1, v2, 0);
  freeexps(fs, e1, e2);
  e1->u.info = pc;
  e1->k = VRELOC;
  luaK_fixline(fs, line);
  luaK_codeABCk(fs, mmop, v1, v2, event, flip);
  luaK_fixline(fs, line);
}
