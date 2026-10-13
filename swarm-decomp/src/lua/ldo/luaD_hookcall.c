#include "lstate.h"
#include "LClosure.h"
#include "Proto.h"
#include "fn_luaD_hook.h"
#include "fn_luaD_hookcall.h"

void luaD_hookcall (lua_State *L, CallInfo *ci) {
  L->oldpc = 0;
  if (L->hookmask & (1 << 0)) {
    int event = (ci->callstatus & (1 << 5)) ? 4 : 0;
    Proto *p = ((LClosure *)ci->func->value_.gc)->p;
    ci->u.l.savedpc++;
    luaD_hook(L, event, -1, 1, p->numparams);
    ci->u.l.savedpc--;
  }
}
