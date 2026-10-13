#include "FuncState.h"
#include "expdesc.h"
#include "luaK_exp2K.h"
#include "OpCode.h"
#include "TMS.h"
#include "fn_luaK_exp2anyreg.h"
#include "tonumeral.h"

static void finishbinexpval (FuncState *fs, expdesc *e1, expdesc *e2, int op, int v2, int flip, int line, OpCode mmop, TMS event);

static void codearith (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int flip, int line) {
  if (tonumeral(e2) && luaK_exp2K(fs, e2))
    finishbinexpval(fs, e1, e2, opr + 0x16, e2->u.info, flip, line, 0x30, opr + 6);
  else {
    if (flip) {
      expdesc tmp = *e1;
      *e1 = *e2;
      *e2 = tmp;
    }
    int v2 = luaK_exp2anyreg(fs, e2);
    finishbinexpval(fs, e1, e2, opr + 0x22, v2, 0, line, 0x2e, opr + 6);
  }
}
