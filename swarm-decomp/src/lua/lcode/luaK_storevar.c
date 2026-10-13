#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_code.h"
#include "fn_luaY_nvarstack.h"
#include "exp2reg.h"
#include "codeABRK.h"

void luaK_storevar (FuncState *fs, expdesc *var, expdesc *ex) {
  switch (var->k) {
    case VLOCAL: {
      if (ex->k == VNONRELOC) {
        if (ex->u.info >= luaY_nvarstack(fs))
          fs->freereg--;
      }
      exp2reg(fs, ex, var->u.var.ridx);
      return;
    }
    case VUPVAL: {
      int e = luaK_exp2anyreg(fs, ex);
      luaK_code(fs, (e << 7) | (var->u.info << 16) | 10);
      break;
    }
    case VINDEXUP: {
      codeABRK(fs, 0xf, var->u.ind.t, var->u.ind.idx, ex);
      break;
    }
    case VINDEXI: {
      codeABRK(fs, 0x11, var->u.ind.t, var->u.ind.idx, ex);
      break;
    }
    case VINDEXSTR: {
      codeABRK(fs, 0x12, var->u.ind.t, var->u.ind.idx, ex);
      break;
    }
    case VINDEXED: {
      codeABRK(fs, 0x10, var->u.ind.t, var->u.ind.idx, ex);
      break;
    }
    default:
      break;
  }
  if (ex->k == VNONRELOC) {
    if (ex->u.info >= luaY_nvarstack(fs))
      fs->freereg--;
  }
}
