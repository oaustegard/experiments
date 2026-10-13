#include "FuncState.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaK_fixline.h"

static void fixforjump (FuncState *fs, int pc, int dest, int back) {
  Instruction *jmp = &fs->f->code[pc];
  int offset = dest - (pc + 1);
  if (back)
    offset = -offset;
  if (offset > 0x1ffff)
    luaX_syntaxerror(fs->ls, "control structure too long");
  *jmp = (*jmp & 0x7fff) | (offset << 15);
}
