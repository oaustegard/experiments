#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_codeABCk.h"
#include "removevalues.h"
#include "freeexp.h"
#include "discharge2anyreg.h"
#include "negatecondition.h"

static void codenot (FuncState *fs, expdesc *e) {
  switch (e->k) {
    case VNIL: case VFALSE: {
      e->k = VTRUE;
      break;
    }
    case VK: case VKFLT: case VKINT: case VKSTR: case VTRUE: {
      e->k = VFALSE;
      break;
    }
    case VJMP: {
      negatecondition(fs, e);
      break;
    }
    case VRELOC: case VNONRELOC: {
      discharge2anyreg(fs, e);
      freeexp(fs, e);
      e->u.info = luaK_codeABCk(fs, 51/* OP_NOT */, 0, e->u.info, 0, 0);
      e->k = VRELOC;
      break;
    }
    default: break;
  }
  {
    int temp = e->f;
    e->f = e->t;
    e->t = temp;
  }
  removevalues(fs, e->f);
  removevalues(fs, e->t);
}
