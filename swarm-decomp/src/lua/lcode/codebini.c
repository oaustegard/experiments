#include "codebini.h"
#include "lcode_ops.h"
#include "finishbinexpval.h"

static void codebini (FuncState *fs, int op, expdesc *e1, expdesc *e2, int flip, int line, TMS event) {
  finishbinexpval(fs, e1, e2, op, (int)e2->u.ival + 0x7f, flip, line, OPC_MMBINI, event);
}
