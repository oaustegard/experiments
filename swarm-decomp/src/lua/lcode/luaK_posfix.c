#include "FuncState.h"
#include "expdesc.h"
#include "BinOpr.h"
#include "OpCode.h"
#include "TMS.h"
#include "Instruction.h"
#include "CREATE_ABCk.h"
#include "lcode_limits.h"
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

/* opcode numbers (lopcodes.h order) */
#define OPC_EQ 57
#define OPC_LT 58
#define OPC_LE 59
#define OPC_EQK 60
#define OPC_EQI 61
#define OPC_LTI 62
#define OPC_CONCAT 53
#define OPC_ADDI 0x15
#define OPC_ADDK 0x16
#define OPC_ADD 0x22
#define OPC_SHRI 0x20
#define OPC_SHLI 0x21
#define OPC_MMBIN 0x2e
#define OPC_MMBINI 0x2f
#define OPC_MMBINK 0x30

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
        Instruction ins;
        if (e2->k == VNONRELOC) {
          if (e2->u.info >= luaY_nvarstack(fs)) fs->freereg--;
        }
        ins = fs->f->code[fs->pc - 1];
        ins = (ins & 0xff00807f) | ((Instruction)e1->u.info << 7 & 0x7f80)
              | (((ins >> 16) + 1) << 16 & 0xff0000);
        fs->f->code[fs->pc - 1] = ins;
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
    case OPR_ADD: {
      int flip = 0;
      if (e1->t == e1->f && (e1->k == VKFLT || e1->k == VKINT)) {
        expdesc tmp = *e1; *e1 = *e2; *e2 = tmp;
        flip = 1;
      }
      if (e2->k == VKINT && e2->t == e2->f && (lua_Unsigned)e2->u.ival + 0x7f <= 0xff)
        finishbinexpval(fs, e1, e2, OPC_ADDI, (int)e2->u.ival + 0x7f, flip, line, OPC_MMBINI, TM_ADD);
      else
        codearith(fs, opr, e1, e2, flip, line);
      break;
    }
    case OPR_MUL: {
      if (e1->k == VKINT) {
        expdesc tmp = *e1; *e1 = *e2; *e2 = tmp;
        if (luaK_exp2K(fs, e2)) {
          finishbinexpval(fs, e1, e2, OPC_ADDK + 2, e2->u.info, 1, line, OPC_MMBINK, TM_MUL);
          break;
        }
        tmp = *e1; *e1 = *e2; *e2 = tmp;
      }
      {
        int v2 = luaK_exp2anyreg(fs, e2);
        finishbinexpval(fs, e1, e2, OPC_ADD + 2, v2, 0, line, OPC_MMBIN, TM_MUL);
      }
      break;
    }
    case OPR_SUB: {
      if (finishbinexpneg(fs, e1, e2, OPC_ADDI, line, TM_SUB))
        break;
      codearith(fs, opr, e1, e2, 0, line);
      break;
    }
    case OPR_MOD: case OPR_POW: case OPR_DIV: case OPR_IDIV:
    case OPR_BAND: case OPR_BOR: case OPR_BXOR: {
      codearith(fs, opr, e1, e2, 0, line);
      break;
    }
    case OPR_SHL: {
      if (e1->k == VKINT && e1->t == e1->f && (lua_Unsigned)e1->u.ival + 0x7f <= 0xff) {
        int v2 = (int)e1->u.ival + 0x7f;
        expdesc tmp = *e1; *e1 = *e2; *e2 = tmp;
        finishbinexpval(fs, e1, e2, OPC_SHLI, v2, 1, line, OPC_MMBINI, TM_SHL);
      }
      else if (finishbinexpneg(fs, e1, e2, OPC_SHRI, line, TM_SHL))
        break;
      else {
        int v2 = luaK_exp2anyreg(fs, e2);
        finishbinexpval(fs, e1, e2, OPC_ADD + 10, v2, 0, line, OPC_MMBIN, TM_SHL);
      }
      break;
    }
    case OPR_SHR: {
      if (e2->k == VKINT && e2->t == e2->f && (lua_Unsigned)e2->u.ival + 0x7f <= 0xff)
        finishbinexpval(fs, e1, e2, OPC_SHRI, (int)e2->u.ival + 0x7f, 0, line, OPC_MMBINI, TM_SHR);
      else {
        int v2 = luaK_exp2anyreg(fs, e2);
        finishbinexpval(fs, e1, e2, OPC_ADD + 11, v2, 0, line, OPC_MMBIN, TM_SHR);
      }
      break;
    }
    case OPR_EQ: case OPR_NE: {
      int pi, isfloat = 0, r1, r2;
      int op;
      if (e1->k != VNONRELOC) {
        expdesc tmp = *e1; *e1 = *e2; *e2 = tmp;
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
        isfloat = 0;
      }
      freeexps(fs, e1, e2);
      codeeq_jmp(fs, e1, CREATE_ABCk(op, r1, r2, isfloat, opr == OPR_EQ));
      break;
    }
    case OPR_LT: case OPR_LE: case OPR_GT: case OPR_GE: {
      int pi, isfloat = 0, r1, r2;
      int op = opr;
      if (opr == OPR_GT || opr == OPR_GE) {
        expdesc tmp = *e1; *e1 = *e2; *e2 = tmp;
        op -= 3;
      }
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
