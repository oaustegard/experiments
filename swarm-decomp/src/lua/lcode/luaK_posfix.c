#include "FuncState.h"
#include "expdesc.h"
#include "BinOpr.h"
#include "OpCode.h"
#include "TMS.h"
#include "Instruction.h"
#include "CREATE_ABCk.h"
#include "lcode_limits.h"
#include "lcode_ops.h"
#include "fn_luaK_concat.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_code.h"
#include "fn_luaK_fixline.h"
#include "fn_luaY_nvarstack.h"
#include "luaK_exp2K.h"
#include "isSCnumber.h"
#include "freeexps.h"
#include "constfolding.h"
#include "codearith.h"
#include "finishbinexpval.h"
#include "finishbinexpneg.h"
#include "codeeq_jmp.h"
#include "swapexps.h"
#include "codecommutative.h"
#include "codebitwise.h"
#include "codebinexpval.h"
#include "codebini.h"
#include "isSCint.h"

void luaK_posfix (FuncState *fs, BinOpr opr, expdesc *e1, expdesc *e2, int line) {
  luaK_dischargevars(fs, e2);
  if (opr <= OPR_SHR && constfolding(fs, opr, e1, e2))
    return;
  switch (opr) {
    case OPR_AND: {
      luaK_concat(fs, &e2->f, e1->f);
      *e1 = *e2;
      break;
    }
    case OPR_OR: {
      luaK_concat(fs, &e2->t, e1->t);
      *e1 = *e2;
      break;
    }
    case OPR_CONCAT: {
      luaK_exp2nextreg(fs, e2);
      if (fs->pc > fs->lasttarget && (fs->f->code[fs->pc - 1] & 0x7f) == OPC_CONCAT) {
        Instruction *ie2 = &fs->f->code[fs->pc - 1];
        int n = (*ie2 >> 16) & 0xff;
        if (e2->k == VNONRELOC) {
          if (e2->u.info >= luaY_nvarstack(fs)) fs->freereg--;
        }
        *ie2 = (*ie2 & ~0x7f80u) | ((Instruction)e1->u.info << 7 & 0x7f80);
        *ie2 = (*ie2 & ~0xff0000u) | ((Instruction)(n + 1) << 16 & 0xff0000);
      }
      else {
        luaK_code(fs, ((Instruction)e1->u.info << 7) | 0x20035);
        if (e2->k == VNONRELOC) {
          if (e2->u.info >= luaY_nvarstack(fs)) fs->freereg--;
        }
        luaK_fixline(fs, line);
      }
      break;
    }
    case OPR_ADD: case OPR_MUL: {
      codecommutative(fs, opr, e1, e2, line);
      break;
    }
    case OPR_SUB: {
      if (finishbinexpneg(fs, e1, e2, OPC_ADDI, line, TM_SUB))
        break;
    }
    /* FALLTHROUGH */
    case OPR_MOD: case OPR_POW: case OPR_DIV: case OPR_IDIV: {
      codearith(fs, opr, e1, e2, 0, line);
      break;
    }
    case OPR_BAND: case OPR_BOR: case OPR_BXOR: {
      codebitwise(fs, opr, e1, e2, line);
      break;
    }
    case OPR_SHL: {
      if (isSCint(e1)) {
        swapexps(e1, e2);
        codebini(fs, OPC_SHLI, e1, e2, 1, line, TM_SHL);
      }
      else if (finishbinexpneg(fs, e1, e2, OPC_SHRI, line, TM_SHL))
        break;
      else
        codebinexpval(fs, opr, e1, e2, line);
      break;
    }
    case OPR_SHR: {
      if (isSCint(e2))
        codebini(fs, OPC_SHRI, e1, e2, 0, line, TM_SHR);
      else
        codebinexpval(fs, opr, e1, e2, line);
      break;
    }
    case OPR_EQ: case OPR_NE: {
      int pi, isfloat = 0, r1, r2;
      int op;
      if (e1->k != VNONRELOC) {
        swapexps(e1, e2);
      }
      r1 = luaK_exp2anyreg(fs, e1);
      if (isSCnumber(e2, &pi, &isfloat)) {
        op = OPC_EQI;
        r2 = pi;
      }
      else if (luaK_exp2K(fs, e2)) {
        op = OPC_EQK;
        r2 = e2->u.info;
      }
      else {
        luaK_exp2anyreg(fs, e2);
        op = OPC_EQ;
        r2 = luaK_exp2anyreg(fs, e2);
      }
      freeexps(fs, e1, e2);
      codeeq_jmp(fs, e1, CREATE_ABCk(op, r1, r2, isfloat, opr == OPR_EQ));
      break;
    }
    case OPR_GT: case OPR_GE: {
      swapexps(e1, e2);
      opr -= 3;
    }
    /* FALLTHROUGH */
    case OPR_LT: case OPR_LE: {
      int pi, isfloat = 0, r1, r2;
      int op = opr;
      if (isSCnumber(e2, &pi, &isfloat)) {
        r1 = luaK_exp2anyreg(fs, e1);
        r2 = pi;
        op += 0x30;
      }
      else if (isSCnumber(e1, &pi, &isfloat)) {
        r1 = luaK_exp2anyreg(fs, e2);
        r2 = pi;
        op += 0x32;
      }
      else {
        r1 = luaK_exp2anyreg(fs, e1);
        r2 = luaK_exp2anyreg(fs, e2);
        op += 0x2c;
      }
      freeexps(fs, e1, e2);
      codeeq_jmp(fs, e1, CREATE_ABCk(op, r1, r2, isfloat, 1));
      break;
    }
    default:
      break;
  }
}
