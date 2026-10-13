#include "lstate.h"
#include "unroll.h"
#include "fn_luaD_poscall.h"
#include "fn_luaV_execute.h"

#define isLua(ci) (!((ci)->callstatus & 2))

static void resume (lua_State *L, void *ud) {
  int n = *((int *)ud);
  StkId firstArg = L->top - n;
  CallInfo *ci = L->ci;
  if (L->status != 0) {
    L->status = 0;
    if (isLua(ci)) {
      ci->u.l.savedpc--;
      L->top = firstArg;
      luaV_execute(L, ci);
    }
    else {
      if (ci->u.c.k != NULL) {
        n = (*ci->u.c.k)(L, 1, ci->u.c.ctx);
      }
      luaD_poscall(L, ci, n);
    }
    unroll(L, NULL);
  }
  else {
    ccall(L, firstArg - 1, -1, 0);
  }
}
