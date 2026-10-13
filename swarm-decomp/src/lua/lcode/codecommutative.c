#include "codecommutative.h"
#include "lcode_ops.h"
#include "TMS.h"
#include "swapexps.h"
#include "codebini.h"
#include "codearith.h"
#include "tonumeral.h"

static void codecommutative (FuncState *fs, int opr, expdesc *e1, expdesc *e2, int line) {
  int flip = 0;
  if (tonumeral(e1)) {
    swapexps(e1, e2);
    flip = 1;
  }
  if (opr == OPR_ADD && e2->k == VKINT && e2->t == e2->f && (lua_Unsigned)e2->u.ival + 0x7f <= 0xff)
    codebini(fs, OPC_ADDI, e1, e2, flip, line, TM_ADD);
  else
    codearith(fs, opr, e1, e2, flip, line);
}
