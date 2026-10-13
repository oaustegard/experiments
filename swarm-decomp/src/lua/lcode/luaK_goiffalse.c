#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_concat.h"
#include "patchlistaux.h"
#include "jumponcond.h"
#include "lcode_limits.h"

void luaK_goiffalse (FuncState *fs, expdesc *e) {
  int pc;
  luaK_dischargevars(fs, e);
  switch (e->k) {
    case VJMP: {
      pc = e->u.info;
      break;
    }
    case VNIL: case VFALSE: {
      pc = -1;
      break;
    }
    default: {
      pc = jumponcond(fs, e, 1);
      break;
    }
  }
  luaK_concat(fs, &e->t, pc);
  {
    int target = fs->pc;
    fs->lasttarget = target;
    patchlistaux(fs, e->f, target, 0xff, target);
  }
  e->f = -1;
}
