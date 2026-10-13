#include "FuncState.h"
#include "expdesc.h"
#include "getjumpcontrol.h"
#include "negatecondition.h"

static void negatecondition (FuncState *fs, expdesc *e) {
  Instruction *pc = getjumpcontrol(fs, e->u.info);
  *pc ^= 0x8000;
}
