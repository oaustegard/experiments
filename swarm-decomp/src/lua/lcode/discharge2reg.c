#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_nil.h"
#include "fn_luaK_code.h"
#include "fn_luaK_int.h"
#include "fn_luaV_flttointeger.h"
#include "luaK_numberK.h"
#include "luaK_codek.h"
#include "addk.h"
#include "lopcodes.h"

static void discharge2reg (FuncState *fs, expdesc *e, int reg) {
  luaK_dischargevars(fs, e);
  switch (e->k) {
    case VNIL: {
      luaK_nil(fs, reg, 1);
      break;
    }
    case VFALSE: {
      luaK_code(fs, (reg << 7) | 5);
      break;
    }
    case VTRUE: {
      luaK_code(fs, (reg << 7) | 7);
      break;
    }
    case VKSTR: {
      TValue o;
      o.value_.gc = (GCObject *)e->u.strval;
      o.tt_ = ((GCObject *)e->u.strval)->tt | 0x40;
      e->u.info = addk(fs, &o, &o);
      e->k = VK;
    }
    /* fall through */
    case VK: {
      luaK_codek(fs, reg, e->u.info);
      break;
    }
    case VKFLT: {
      luaK_float(fs, reg, e->u.nval);
      break;
    }
    case VKINT: {
      luaK_int(fs, reg, e->u.ival);
      break;
    }
    case VRELOC: {
      Instruction *pc = &fs->f->code[e->u.info];
      *pc = (*pc & ~(0xffu << 7)) | ((reg << 7) & 0x7f80);
      break;
    }
    case VNONRELOC: {
      if (reg != e->u.info)
        luaK_code(fs, (reg << 7) | (e->u.info << 16));
      break;
    }
    default: {
      return;
    }
  }
  e->u.info = reg;
  e->k = VNONRELOC;
}
