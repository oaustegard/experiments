#include "FuncState.h"
#include "expdesc.h"
#include "BinOpr.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_goiftrue.h"
#include "fn_luaK_goiffalse.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_exp2anyreg.h"
#include "luaK_exp2K.h"
#include "isSCnumber.h"
#include "tonumeral.h"

void luaK_infix (FuncState *fs, BinOpr op, expdesc *v) {
  luaK_dischargevars(fs, v);
  switch (op) {
    case OPR_AND: {
      luaK_goiftrue(fs, v);
      break;
    }
    case OPR_OR: {
      luaK_goiffalse(fs, v);
      break;
    }
    case OPR_CONCAT: {
      luaK_exp2nextreg(fs, v);
      break;
    }
    case OPR_ADD: case OPR_SUB:
    case OPR_MUL: case OPR_DIV: case OPR_IDIV:
    case OPR_MOD: case OPR_POW:
    case OPR_BAND: case OPR_BOR: case OPR_BXOR:
    case OPR_SHL: case OPR_SHR: {
      if (!tonumeral(v))
        luaK_exp2anyreg(fs, v);
      break;
    }
    case OPR_EQ: case OPR_NE: {
      if (!tonumeral(v))
        if (!luaK_exp2K(fs, v))
          luaK_exp2anyreg(fs, v);
      break;
    }
    case OPR_LT: case OPR_LE:
    case OPR_GT: case OPR_GE: {
      int dummy, dummy2;
      if (!isSCnumber(v, &dummy, &dummy2))
        luaK_exp2anyreg(fs, v);
      break;
    }
    default: break;
  }
}
