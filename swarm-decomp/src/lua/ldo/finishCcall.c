#include "lstate.h"
#include "fn_luaD_poscall.h"
#include "finishpcallk.h"

static void finishCcall (lua_State *L, CallInfo *ci) {
  int n;
  if (ci->callstatus & 0x200) {
    n = ci->u2.nres;
  }
  else {
    int status = 1;
    if (ci->callstatus & 0x10)
      status = finishpcallk(L, ci);
    if (L->ci->top < L->top)
      L->ci->top = L->top;
    n = (*ci->u.c.k)(L, status, ci->u.c.ctx);
  }
  luaD_poscall(L, ci, n);
}
