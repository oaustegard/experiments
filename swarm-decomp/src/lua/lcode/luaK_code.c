#include "fn_luaK_code.h"
#include "savelineinfo.h"
#include "lcode_limits.h"
#include "fn_luaM_growaux_.h"

int luaK_code (FuncState *fs, Instruction i) {
  Proto *f = fs->f;
  f->code = luaM_growaux_(fs->ls->L, f->code, fs->pc, &f->sizecode,
                          sizeof(Instruction), MAX_INT, "opcodes");
  f->code[fs->pc++] = i;
  savelineinfo(fs, f, fs->ls->lastline);
  return fs->pc - 1;
}
