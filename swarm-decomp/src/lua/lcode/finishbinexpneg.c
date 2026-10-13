#include "FuncState.h"
#include "expdesc.h"
#include "lcode_limits.h"
#include "OpCode.h"
#include "TMS.h"

static void finishbinexpval (FuncState *fs, expdesc *e1, expdesc *e2, int op, int v2, int flip, int line, OpCode mmop, TMS event);

static int finishbinexpneg (FuncState *fs, expdesc *e1, expdesc *e2, int op, int line, int event) {
  if (e2->k == VKINT && e2->f == e2->t) {
    lua_Integer i2 = e2->u.ival;
    if ((unsigned long)(i2 + 0x7f) <= 0xff && (unsigned long)(0x7f - i2) <= 0xff) {
      int v2 = (int)i2;
      finishbinexpval(fs, e1, e2, op, 0x7f - v2, 0, line, 0x2f, event);
      fs->f->code[fs->pc - 1] = (fs->f->code[fs->pc - 1] & 0xff00ffff) | ((unsigned)(v2 + 0x7f) << 16);
      return 1;
    }
  }
  return 0;
}
