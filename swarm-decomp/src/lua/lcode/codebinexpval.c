#include "codebinexpval.h"
#include "lcode_ops.h"
#include "TMS.h"
#include "fn_luaK_exp2anyreg.h"
#include "finishbinexpval.h"

static void codebinexpval (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int line) {
  int v2 = luaK_exp2anyreg(fs, e2);
  finishbinexpval(fs, e1, e2, OPC_ADD + opr, v2, 0, line, OPC_MMBIN, opr + 6);
}
