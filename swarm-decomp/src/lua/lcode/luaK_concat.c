#include "FuncState.h"
#include "fn_luaK_concat.h"
#include "fn_luaX_syntaxerror.h"
#include "getjump.h"

void luaK_concat (FuncState *fs, int *l1, int l2) {
  if (l2 == -1)
    return;
  else if (*l1 == -1)
    *l1 = l2;
  else {
    int list = *l1;
    int next;
    while ((next = getjump(fs, list)) != -1)
      list = next;
    {
      Instruction *jmp = &fs->f->code[list];
      int offset = l2 - (list + 1) + 0xffffff;
      if ((unsigned)offset > 0x1ffffff)
        luaX_syntaxerror(fs->ls, "control structure too long");
      *jmp = (*jmp & 0x7f) | ((unsigned)offset << 7);
    }
  }
}
