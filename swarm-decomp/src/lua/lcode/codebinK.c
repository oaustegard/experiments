#include "codebinK.h"
#include "lcode_ops.h"
#include "TMS.h"
#include "finishbinexpval.h"

static void codebinK (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int flip, int line) {
  int v2 = e2->u.info;
  finishbinexpval(fs, e1, e2, OPC_ADDK + opr, v2, flip, line, OPC_MMBINK, opr + 6);
}
