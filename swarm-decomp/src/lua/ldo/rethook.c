#include "lstate.h"
#include "fn_luaD_hook.h"
#include "rethook.h"

#define isLua(ci) (!((ci)->callstatus & 2))
#define ci_func(ci) ((LClosure *)((ci)->func->value_.gc))
#define pcRel(pc, p) ((int)((pc) - (p)->code) - 1)

static void rethook (lua_State *L, CallInfo *ci, int nres) {
  if (L->hookmask & (1 << 1)) {
    StkId firstres = L->top - nres;
    int delta = 0;
    int ftransfer;
    if (isLua(ci)) {
      Proto *p = ci_func(ci)->p;
      if (p->is_vararg)
        delta = ci->u.l.nextraargs + p->numparams + 1;
    }
    ci->func += delta;
    ftransfer = (unsigned short)(firstres - ci->func);
    luaD_hook(L, 1, -1, ftransfer, nres);
    ci->func -= delta;
  }
  if (isLua(ci = ci->previous))
    L->oldpc = pcRel(ci->u.l.savedpc, ci_func(ci)->p);
}
