#include "FuncState.h"
#include "expdesc.h"
#include "Dyndata.h"
#include "LexState.h"
#include "fn_luaK_code.h"
#include "fn_luaK_setoneret.h"
#include "fn_luaK_dischargevars.h"
#include "CREATE_ABCk.h"
#include "freereg.h"
#include "freeregs.h"
#include "const2exp.h"

void luaK_dischargevars (FuncState *fs, expdesc *e) {
  switch (e->k) {
    case VCONST: {
      const2exp(&fs->ls->dyd->actvar.arr[e->u.info].k, e);
      break;
    }
    case VLOCAL: {
      e->u.info = e->u.var.ridx;
      e->k = VNONRELOC;
      break;
    }
    case VUPVAL: {
      e->u.info = luaK_code(fs, CREATE_ABCk(9, 0, e->u.info, 0, 0));
      e->k = VRELOC;
      break;
    }
    case VINDEXUP: {
      e->u.info = luaK_code(fs, CREATE_ABCk(11, 0, e->u.ind.t, e->u.ind.idx, 0));
      e->k = VRELOC;
      break;
    }
    case VINDEXI: {
      freereg(fs, e->u.ind.t);
      e->u.info = luaK_code(fs, CREATE_ABCk(13, 0, e->u.ind.t, e->u.ind.idx, 0));
      e->k = VRELOC;
      break;
    }
    case VINDEXSTR: {
      freereg(fs, e->u.ind.t);
      e->u.info = luaK_code(fs, CREATE_ABCk(14, 0, e->u.ind.t, e->u.ind.idx, 0));
      e->k = VRELOC;
      break;
    }
    case VINDEXED: {
      freeregs(fs, e->u.ind.t, e->u.ind.idx);
      e->u.info = luaK_code(fs, CREATE_ABCk(12, 0, e->u.ind.t, e->u.ind.idx, 0));
      e->k = VRELOC;
      break;
    }
    case VVARARG:
    case VCALL: {
      luaK_setoneret(fs, e);
      break;
    }
    default:
      break;
  }
}
