#include "FuncState.h"
#include "Proto.h"
#include "OpCode.h"
#include "fn_luaX_syntaxerror.h"

void luaK_finish (FuncState *fs) {
  int i;
  Proto *p = fs->f;
  for (i = 0; i < fs->pc; i++) {
    Instruction *pc = &p->code[i];
    switch (*pc & 0x7f) {
      case 71: case 72: {  /* OP_RETURN0, OP_RETURN1 */
        if (!(fs->needclose || p->is_vararg))
          break;  /* no extra work */
        /* else use OP_RETURN with extra arguments */
        *pc = (*pc & 0xffffff80) | 70;
      }  /* FALLTHROUGH */
      case 69: case 70: {  /* OP_TAILCALL, OP_RETURN */
        if (fs->needclose)
          *pc |= 0x8000;  /* signal that it needs to close */
        if (p->is_vararg)
          *pc = (*pc & 0x00ffffff) | ((Instruction)(p->numparams + 1) << 24);
        break;
      }
      case 56: {  /* OP_JMP */
        int target = finaltarget(p->code, i);
        fixjump(fs, i, target);
        break;
      }
      default: break;
    }
  }
}
