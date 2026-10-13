#include "FuncState.h"
#include "expdesc.h"
#include "Instruction.h"
#include "fn_luaK_code.h"

static void codeeq_jmp (FuncState *fs, expdesc *e1, Instruction ins) {
  luaK_code(fs, ins);
  e1->u.info = luaK_code(fs, 0x7fffff38);  /* JMP to be patched */
  e1->k = VJMP;
}
