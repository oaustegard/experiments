#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_concat.h"
#include "patchlistaux.h"
#include "jumponcond.h"
#include "lcode_limits.h"
#include "negatecondition.h"

void luaK_goiftrue (FuncState *fs, expdesc *e) {
  int pc;
  luaK_dischargevars(fs, e);
  switch (e->k) {
    case VJMP: {
      negatecondition(fs, e);
      pc = e->u.info;
      break;
    }
    case VK: case VKFLT: case VKINT: case VKSTR: case VTRUE: {
      pc = -1;
      break;
    }
    default: {
      pc = jumponcond(fs, e, 0);
      break;
    }
  }
  luaK_concat(fs, &e->f, pc);
  {
    int target = fs->pc;
    fs->lasttarget = target;
    patchlistaux(fs, e->t, target, 0xff, target);
  }
  e->t = -1;
}
