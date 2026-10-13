#include "FuncState.h"
#include "expdesc.h"
#include "OpCode.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_code.h"
#include "fn_luaK_fixline.h"
#include "freeexp.h"

static void codeunexpval (FuncState *fs, OpCode op, expdesc *e, int line) {
  int r = luaK_exp2anyreg(fs, e);
  freeexp(fs, e);
  e->u.info = luaK_code(fs, op | (r << 16));
  e->k = VRELOC;
  luaK_fixline(fs, line);
}
