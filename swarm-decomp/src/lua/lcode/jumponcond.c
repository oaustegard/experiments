#include "FuncState.h"
#include "expdesc.h"
#include "CREATE_ABCk.h"
#include "lcode_limits.h"
#include "fn_luaK_code.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaY_nvarstack.h"
#include "discharge2reg.h"
#include "jumponcond.h"

/* opcodes not in lopcodes.h: OP_NOT 0x33, OP_TEST 0x42, OP_TESTSET 0x43, OP_JMP 0x38 */

static int jumponcond (FuncState *fs, expdesc *e, int cond) {
  if (e->k == VRELOC) {
    Instruction ie = fs->f->code[e->u.info];
    if ((ie & 0x7f) == 0x33) {
      /* removelastinstruction */
      int pc = fs->pc - 1;
      if (fs->f->lineinfo[pc] != ABSLINEINFO) {
        fs->previousline -= fs->f->lineinfo[pc];
        fs->iwthabs--;
      }
      else {
        fs->nabslineinfo--;
        fs->iwthabs = MAXIWTHABS + 1;
      }
      fs->pc--;
      luaK_code(fs, CREATE_ABCk(0x42, (ie >> 16) & 0xff, 0, 0, !cond));
      return luaK_code(fs, 0x7fffff38);
    }
  }
  if (e->k != VNONRELOC) {
    luaK_reserveregs(fs, 1);
    discharge2reg(fs, e, fs->freereg - 1);
  }
  if (e->k == VNONRELOC) {
    if (e->u.info >= luaY_nvarstack(fs))
      fs->freereg--;
  }
  luaK_code(fs, CREATE_ABCk(0x43, 0xff, e->u.info, 0, cond));
  return luaK_code(fs, 0x7fffff38);
}
